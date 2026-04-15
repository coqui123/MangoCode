// app.rs — App state struct and main event loop.

use crate::bridge_state::BridgeConnectionState;
use crate::context_viz::ContextVizState;
use crate::dialog_select::{DialogSelectState, SelectItem};
use crate::dialogs::McpApprovalDialogState;
use crate::dialogs::PermissionRequest;
use crate::diff_viewer::{build_turn_diff, DiffViewerState};
use crate::export_dialog::{ExportDialogState, ExportFormat};
use crate::mcp_view::{McpServerView, McpToolView, McpViewState, McpViewStatus};
use crate::model_picker::{EffortLevel, ModelPickerState, FAST_MODE_MODEL};
use crate::notifications::{NotificationKind, NotificationQueue};
use crate::overlays::{
    GlobalSearchState, HelpOverlay, HistorySearchOverlay, MessageSelectorOverlay,
    RewindFlowOverlay, SelectorMessage,
};
use crate::plugin_views::PluginHintBanner;
use crate::privacy_screen::PrivacyScreen;
use crate::prompt_input::{InputMode, PromptInputState, VimMode};
use crate::render;
use crate::session_browser::SessionBrowserState;
use crate::settings_screen::SettingsScreen;
use crate::stats_dialog::StatsDialogState;
use crate::tasks_overlay::TasksOverlay;
use crate::theme_screen::ThemeScreen;
use crate::{
    agents_view::{AgentInfo, AgentStatus, AgentsMenuState, AgentsRoute},
    diff_viewer::DiffPane,
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use mangocode_core::config::{Config, Settings, Theme};
use mangocode_core::cost::CostTracker;
use mangocode_core::file_history::FileHistory;
use mangocode_core::keybindings::{
    KeyContext, KeybindingResolver, KeybindingResult, ParsedKeystroke, UserKeybindings,
};
use mangocode_core::types::{Message, Role};
use mangocode_query::QueryEvent;
use mangocode_tools;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::cell::{Cell, RefCell};
use std::io::Stdout;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::debug;
use crate::slash_commands::{prompt_slash_commands, SlashCommandSpec};

const PASTE_BURST_PRINTABLE_GAP: Duration = Duration::from_millis(120);
const PASTE_BURST_IDLE_TIMEOUT: Duration = Duration::from_millis(1500);
const PASTE_BURST_MIN_STREAK: usize = 4;

// ---------------------------------------------------------------------------
// Provider connection helpers
// ---------------------------------------------------------------------------

/// Return the environment variable name for a given provider ID.
fn get_env_var_for_provider(id: &str) -> &'static str {
    match id {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "google" => "GOOGLE_API_KEY",
        "google-vertex" => "VERTEX_PROJECT_ID",
        "github-copilot" => "GITHUB_TOKEN",
        "groq" => "GROQ_API_KEY",
        "cerebras" => "CEREBRAS_API_KEY",
        "sambanova" => "SAMBANOVA_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "togetherai" => "TOGETHER_API_KEY",
        "perplexity" => "PERPLEXITY_API_KEY",
        "cohere" => "COHERE_API_KEY",
        "xai" => "XAI_API_KEY",
        "deepinfra" => "DEEPINFRA_API_KEY",
        "azure" => "AZURE_API_KEY",
        "amazon-bedrock" => "AWS_ACCESS_KEY_ID",
        "sap-ai-core" => "AICORE_SERVICE_KEY",
        "gitlab" => "GITLAB_TOKEN",
        "cloudflare-ai-gateway" | "cloudflare-workers-ai" => "CLOUDFLARE_API_TOKEN",
        "vercel" => "AI_GATEWAY_API_KEY",
        "helicone" => "HELICONE_API_KEY",
        "huggingface" => "HF_TOKEN",
        "nvidia" => "NVIDIA_API_KEY",
        "alibaba" => "DASHSCOPE_API_KEY",
        "venice" => "VENICE_API_KEY",
        "moonshotai" => "MOONSHOT_API_KEY",
        "zhipuai" => "ZHIPU_API_KEY",
        "siliconflow" => "SILICONFLOW_API_KEY",
        "nebius" => "NEBIUS_API_KEY",
        "novita" => "NOVITA_API_KEY",
        "minimax" => "MINIMAX_API_KEY",
        "ovhcloud" => "OVHCLOUD_API_KEY",
        "scaleway" => "SCALEWAY_API_KEY",
        "vultr" => "VULTR_API_KEY",
        "baseten" => "BASETEN_API_KEY",
        "friendli" => "FRIENDLI_TOKEN",
        "upstage" => "UPSTAGE_API_KEY",
        "stepfun" => "STEPFUN_API_KEY",
        "fireworks" => "FIREWORKS_API_KEY",
        _ => "API_KEY",
    }
}

/// Return a URL hint for obtaining an API key from a given provider.
fn get_url_for_provider(id: &str) -> &'static str {
    match id {
        "anthropic" => "console.anthropic.com",
        "openai" => "platform.openai.com/api-keys",
        "google" => "aistudio.google.com/apikey",
        "github-copilot" => "github.com/settings/tokens",
        "groq" => "console.groq.com/keys",
        "cerebras" => "cloud.cerebras.ai",
        "sambanova" => "cloud.sambanova.ai",
        "deepseek" => "platform.deepseek.com/api_keys",
        "mistral" => "console.mistral.ai/api-keys",
        "openrouter" => "openrouter.ai/keys",
        "togetherai" => "api.together.xyz/settings/api-keys",
        "perplexity" => "perplexity.ai/settings/api",
        "cohere" => "dashboard.cohere.com/api-keys",
        "xai" => "console.x.ai",
        "deepinfra" => "deepinfra.com/dash/api_keys",
        "azure" => "portal.azure.com",
        "amazon-bedrock" => "console.aws.amazon.com/bedrock",
        "minimax" => "platform.minimaxi.com",
        "huggingface" => "huggingface.co/settings/tokens",
        "nvidia" => "build.nvidia.com",
        "venice" => "venice.ai/settings/api",
        _ => "the provider's website",
    }
}

/// Validate basic credential shape before persisting.
/// Returns `(category, detail)` on validation failure.
fn validate_provider_credential(
    provider_id: &str,
    key: &str,
) -> Result<(), (&'static str, String)> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err((
            "Configuration",
            format!(
                "Missing credential for {}. Set {} or visit {}.",
                provider_id,
                get_env_var_for_provider(provider_id),
                get_url_for_provider(provider_id)
            ),
        ));
    }

    if trimmed.len() < 10 {
        return Err((
            "Authentication",
            "Credential looks too short to be valid.".to_string(),
        ));
    }

    let prefix_hint = match provider_id {
        "anthropic" if !trimmed.starts_with("sk-ant-") => {
            Some("Anthropic keys usually start with 'sk-ant-'.")
        }
        "openai" if !(trimmed.starts_with("sk-") || trimmed.starts_with("sess-")) => {
            Some("OpenAI keys usually start with 'sk-' or 'sess-'.")
        }
        "google" if !trimmed.starts_with("AIza") => {
            Some("Google Gemini API keys from AI Studio usually start with 'AIza'.")
        }
        "groq" if !trimmed.starts_with("gsk_") => Some("Groq keys usually start with 'gsk_'."),
        _ => None,
    };

    if let Some(hint) = prefix_hint {
        return Err(("Authentication", hint.to_string()));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Visual style for inline system messages in the conversation pane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemMessageStyle {
    Info,
    Warning,
    /// Compact / auto-compact boundary marker.
    Compact,
}

/// A synthetic system annotation inserted between conversation messages.
/// `after_index` is the index in `App::messages` after which this annotation
/// should appear (0 = before all messages, 1 = after message 0, etc.).
#[derive(Debug, Clone)]
pub struct SystemAnnotation {
    pub after_index: usize,
    pub text: String,
    pub style: SystemMessageStyle,
}

/// A displayable item in the conversation pane — either a real message or
/// a synthetic system annotation (e.g. compact boundary).
/// Used only by `render.rs`; constructed on the fly from `messages` +
/// `system_annotations`.
#[derive(Debug, Clone)]
pub enum DisplayMessage {
    /// A real conversation turn.
    Conversation(Message),
    /// An injected system notice (e.g. compact boundary).
    System {
        text: String,
        style: SystemMessageStyle,
    },
}

/// Context menu state: position and currently selected item index.
#[derive(Debug, Clone, Copy)]
pub struct ContextMenuState {
    /// X coordinate of the menu (column).
    pub x: u16,
    /// Y coordinate of the menu (row).
    pub y: u16,
    /// Currently selected menu item index (0-based).
    pub selected_index: usize,
    /// What the context menu is acting on.
    pub kind: ContextMenuKind,
}

/// What content the context menu is currently targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMenuKind {
    /// A specific transcript message.
    Message { message_index: usize },
    /// The current text selection anywhere in the frame.
    Selection,
}

/// Available context menu items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMenuItem {
    Copy,
    Fork,
}

/// State for the Go to Line dialog (Ctrl+G in message pane).
#[derive(Debug, Clone)]
pub struct GoToLineDialog {
    /// Input field for line number.
    pub input: String,
    /// Whether the dialog is currently active.
    pub active: bool,
    /// Total number of lines (for validation feedback).
    pub total_lines: usize,
}

impl Default for GoToLineDialog {
    fn default() -> Self {
        Self::new()
    }
}

impl GoToLineDialog {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            active: false,
            total_lines: 0,
        }
    }

    pub fn open(&mut self, total_lines: usize) {
        self.input.clear();
        self.active = true;
        self.total_lines = total_lines;
    }

    pub fn close(&mut self) {
        self.active = false;
        self.input.clear();
    }

    /// Parse the input as a line number (1-indexed).
    /// Returns None if invalid or out of range.
    pub fn parse_line_number(&self) -> Option<usize> {
        let line_num: usize = self.input.trim().parse().ok()?;
        if line_num >= 1 && line_num <= self.total_lines {
            Some(line_num)
        } else {
            None
        }
    }
}

/// Status of an active or completed tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Done,
    Error,
}

/// Represents an active or completed tool invocation visible in the UI.
#[derive(Debug, Clone)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub status: ToolStatus,
    pub output_preview: Option<String>,
    /// JSON-serialised input for the tool call (populated from the API stream).
    pub input_json: String,
}

/// State for Ctrl+R history search mode (legacy inline struct, kept for test
/// compatibility — the overlay version lives in `overlays::HistorySearchOverlay`).
#[derive(Debug, Clone)]
pub struct HistorySearch {
    pub query: String,
    /// Indices into `input_history` that match the current query.
    pub matches: Vec<usize>,
    /// Which match is currently highlighted.
    pub selected: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TranscriptSearchState {
    pub query: String,
    pub total_matches: usize,
    pub current_match: Option<usize>,
}

impl Default for HistorySearch {
    fn default() -> Self {
        Self::new()
    }
}

impl HistorySearch {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            matches: Vec::new(),
            selected: 0,
        }
    }

    /// Re-compute matches against the given history slice.
    pub fn update_matches(&mut self, history: &[String]) {
        let q = self.query.to_lowercase();
        self.matches = history
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.to_lowercase().contains(&q) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        // Clamp selected to valid range
        if !self.matches.is_empty() && self.selected >= self.matches.len() {
            self.selected = self.matches.len() - 1;
        }
    }

    /// Return the currently selected history entry, if any.
    pub fn current_entry<'a>(&self, history: &'a [String]) -> Option<&'a str> {
        self.matches
            .get(self.selected)
            .and_then(|&i| history.get(i))
            .map(String::as_str)
    }
}

/// Attempt to copy text to the system clipboard using platform CLI tools.
/// Returns true if successful.
pub fn try_copy_to_clipboard(text: &str) -> bool {
    // Windows
    #[cfg(target_os = "windows")]
    {
        use std::io::Write;
        if let Ok(mut child) = std::process::Command::new("clip")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            return child.wait().map(|s| s.success()).unwrap_or(false);
        }
    }
    // macOS
    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        if let Ok(mut child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            return child.wait().map(|s| s.success()).unwrap_or(false);
        }
    }
    // Linux / Wayland / X11
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        for cmd in &[
            "wl-copy",
            "xclip -selection clipboard",
            "xsel --clipboard --input",
        ] {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if let Some((prog, args)) = parts.split_first() {
                if let Ok(mut child) = std::process::Command::new(prog)
                    .args(args)
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                {
                    if let Some(stdin) = child.stdin.as_mut() {
                        let _ = stdin.write_all(text.as_bytes());
                    }
                    if child.wait().map(|s| s.success()).unwrap_or(false) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn key_event_to_keystroke(key: &KeyEvent) -> Option<ParsedKeystroke> {
    fn layout_to_latin(c: char) -> String {
        let lower = c.to_lowercase().next().unwrap_or(c);
        let mapped: Option<char> = match lower {
            // Russian/Ukrainian JCUKEN -> QWERTY position mapping
            'й' => Some('q'),
            'ц' => Some('w'),
            'у' => Some('e'),
            'к' => Some('r'),
            'е' => Some('t'),
            'н' => Some('y'),
            'г' => Some('u'),
            'ш' => Some('i'),
            'щ' => Some('o'),
            'з' => Some('p'),
            'ф' => Some('a'),
            'ы' => Some('s'),
            'в' => Some('d'),
            'а' => Some('f'),
            'п' => Some('g'),
            'р' => Some('h'),
            'о' => Some('j'),
            'л' => Some('k'),
            'д' => Some('l'),
            'я' => Some('z'),
            'ч' => Some('x'),
            'с' => Some('c'),
            'м' => Some('v'),
            'и' => Some('b'),
            'т' => Some('n'),
            'ь' => Some('m'),
            'і' => Some('s'),
            'ї' => Some(']'),
            'є' => Some('\''),
            _ => None,
        };
        mapped.unwrap_or(lower).to_string()
    }

    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    let normalized_key = match key.code {
        KeyCode::Backspace => "backspace".to_string(),
        KeyCode::Delete => "delete".to_string(),
        KeyCode::Down => "down".to_string(),
        KeyCode::End => "end".to_string(),
        KeyCode::Enter => "enter".to_string(),
        KeyCode::Esc => "escape".to_string(),
        KeyCode::Home => "home".to_string(),
        KeyCode::Left => "left".to_string(),
        KeyCode::PageDown => "pagedown".to_string(),
        KeyCode::PageUp => "pageup".to_string(),
        KeyCode::Right => "right".to_string(),
        KeyCode::Tab => "tab".to_string(),
        KeyCode::Up => "up".to_string(),
        KeyCode::BackTab => "tab".to_string(),
        KeyCode::Char(' ') => "space".to_string(),
        KeyCode::Char(c) => {
            if (ctrl || alt) && !c.is_ascii() {
                layout_to_latin(c)
            } else {
                c.to_lowercase().to_string()
            }
        }
        _ => return None,
    };

    Some(ParsedKeystroke {
        key: normalized_key,
        ctrl,
        alt,
        shift: key.modifiers.contains(KeyModifiers::SHIFT),
        meta: key.modifiers.contains(KeyModifiers::SUPER),
    })
}

// ---------------------------------------------------------------------------
// Focus target
// ---------------------------------------------------------------------------

/// Which area of the TUI currently has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusTarget {
    /// Keyboard input goes to the prompt editor.
    Input,
    /// Keyboard input goes to the transcript/message pane (scroll, etc.).
    Transcript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalOwner {
    PermissionRequest,
    Settings,
    Theme,
    Privacy,
    Stats,
    McpView,
    AgentsMenu,
    DiffViewer,
    BypassPermissions,
    Onboarding,
    DeviceAuth,
    KeyInput,
    Connect,
    InvalidConfig,
    DesktopUpsell,
    MemoryUpdateNotification,
    VoiceModeNotice,
    OverageUpsell,
    HooksConfig,
    MemoryFileSelector,
    FeedbackSurvey,
    GoToLineDialog,
    McpApproval,
    ContextViz,
    ExportDialog,
    SessionBranching,
    SessionBrowser,
    ModelPicker,
    Elicitation,
    CommandPalette,
    DeviceAuthPending,
    GlobalSearch,
    HistorySearch,
    Help,
    Tasks,
    Rewind,
}

// ---------------------------------------------------------------------------
// App struct
// ---------------------------------------------------------------------------

/// The top-level TUI application.
pub struct App {
    // Core state
    pub config: Config,
    pub cost_tracker: Arc<CostTracker>,
    pub messages: Vec<Message>,
    /// Combined display list kept in sync with `messages`: real conversation turns
    /// plus injected system annotations. Used by the renderer so it can iterate
    /// a single sequence instead of merging two lists on every frame.
    pub display_messages: Vec<DisplayMessage>,
    /// Synthetic system annotations interleaved between real messages at render time.
    pub system_annotations: Vec<SystemAnnotation>,
    pub input: String,
    pub prompt_input: PromptInputState,
    pub input_history: Vec<String>,
    pub history_index: Option<usize>,
    pub scroll_offset: usize,
    pub is_streaming: bool,
    pub streaming_text: String,
    pub status_message: Option<String>,
    /// Randomly chosen thinking verb shown next to the spinner while streaming.
    pub spinner_verb: Option<String>,
    pub should_quit: bool,
    pub show_help: bool,

    // Extended state
    pub tool_use_blocks: Vec<ToolUseBlock>,
    pub permission_request: Option<PermissionRequest>,
    pub frame_count: u64,
    pub token_count: u32,
    /// Maximum token budget (from env var or model context window) — P2 feature flag
    pub token_budget: Option<u32>,
    pub cost_usd: f64,
    pub model_name: String,
    /// Whether the app has valid API credentials configured.
    /// False = show the in-TUI provider setup dialog on startup.
    pub has_credentials: bool,
    /// Current effort level (controls extended-thinking budget_tokens).
    pub effort_level: EffortLevel,
    /// Whether fast mode is currently active (model locked to FAST_MODE_MODEL).
    pub fast_mode: bool,
    /// Current agent mode name: "build", "plan", "explore", etc.
    pub agent_mode: Option<String>,
    pub agent_status: Vec<(String, String)>,
    pub history_search: Option<HistorySearch>,
    pub transcript_search: TranscriptSearchState,
    pub keybindings: KeybindingResolver,

    // Cursor position within input (byte offset)
    pub cursor_pos: usize,

    // ---- Scrollback / auto-scroll -----------------------------------------
    /// When `true`, the message pane follows the latest messages automatically.
    pub auto_scroll: bool,
    /// Count of messages that arrived while the user was scrolled up.
    pub new_messages_while_scrolled: usize,

    // ---- Token warning tracking -------------------------------------------
    /// Which threshold (0 = none, 80, 95, 100) was last notified so we only
    /// show each banner once.
    pub token_warning_threshold_shown: u8,

    // ---- Session timing ---------------------------------------------------
    /// Instant the session started (used for elapsed-time in the status bar).
    pub session_start: std::time::Instant,
    /// Instant the current turn's streaming began (reset each time streaming starts).
    pub turn_start: Option<std::time::Instant>,
    /// Elapsed time string for the last completed turn, e.g. "2m 5s".
    pub last_turn_elapsed: Option<String>,
    /// Past-tense verb shown after turn completes, e.g. "Worked" / "Baked".
    pub last_turn_verb: Option<&'static str>,
    /// Incremented whenever transcript-visible state changes so rendering can
    /// reuse cached layout between keystrokes.
    pub transcript_version: Cell<u64>,

    // ---- New overlay / notification fields --------------------------------
    /// Full-screen help overlay (F1 / /help).
    pub help_overlay: HelpOverlay,
    /// Ctrl+R history search overlay.
    pub history_search_overlay: HistorySearchOverlay,
    /// Global ripgrep search / quick-open overlay.
    pub global_search: GlobalSearchState,
    /// Message selector used by /rewind.
    pub message_selector: MessageSelectorOverlay,
    /// Multi-step rewind flow overlay.
    pub rewind_flow: RewindFlowOverlay,
    /// Bridge connection state.
    pub bridge_state: BridgeConnectionState,
    /// Active notification queue.
    pub notifications: NotificationQueue,
    /// Plugin hint banners.
    pub plugin_hints: Vec<PluginHintBanner>,
    /// Optional session title shown in the status bar.
    pub session_title: Option<String>,
    /// Remote session URL (set when bridge connects; readable by commands).
    pub remote_session_url: Option<String>,
    /// Live MCP manager snapshot source when available.
    pub mcp_manager: Option<Arc<mangocode_mcp::McpManager>>,
    /// Queued request for a real MCP reconnect from the interactive loop.
    pub pending_mcp_reconnect: bool,
    /// Shared file-history service used for turn diff reconstruction.
    pub file_history: Option<Arc<parking_lot::Mutex<FileHistory>>>,
    /// Shared query-loop turn counter for turn-local diff reconstruction.
    pub current_turn: Option<Arc<std::sync::atomic::AtomicUsize>>,

    // ---- Visual mode indicators -------------------------------------------
    /// Plan mode — input border turns blue, [PLAN] shown in status bar.
    pub plan_mode: bool,
    /// "While you were away" summary text shown on the welcome screen.
    pub away_summary: Option<String>,
    /// When streaming stalled (used to turn the spinner red after 3 s).
    pub stall_start: Option<std::time::Instant>,

    // ---- Settings / theme / privacy screens --------------------------------
    /// Full-screen tabbed settings screen (/config, /settings).
    pub settings_screen: SettingsScreen,
    /// Theme picker overlay (/theme).
    pub theme_screen: ThemeScreen,
    /// Privacy settings dialog (/privacy-settings).
    pub privacy_screen: PrivacyScreen,
    /// Token/cost analytics dialog.
    pub stats_dialog: StatsDialogState,
    /// MCP server browser and tool detail view.
    pub mcp_view: McpViewState,
    /// Agent definitions and active agent status overlay.
    pub agents_menu: AgentsMenuState,
    /// Diff viewer overlay.
    pub diff_viewer: DiffViewerState,
    /// Session-quality feedback survey overlay.
    pub feedback_survey: crate::feedback_survey::FeedbackSurveyState,
    /// Memory file selector overlay (AGENTS.md browser).
    pub memory_file_selector: crate::memory_file_selector::MemoryFileSelectorState,
    /// Read-only hooks configuration browser.
    pub hooks_config_menu: crate::hooks_config_menu::HooksConfigMenuState,
    /// Overage credit upsell banner.
    pub overage_upsell: crate::overage_upsell::OverageCreditUpsellState,
    /// Voice mode availability notice.
    pub voice_mode_notice: crate::voice_mode_notice::VoiceModeNoticeState,
    /// Desktop app upsell startup dialog.
    pub desktop_upsell: crate::desktop_upsell_startup::DesktopUpsellStartupState,
    /// Startup error dialog for malformed settings.json or AGENTS.md.
    pub invalid_config_dialog: crate::invalid_config_dialog::InvalidConfigDialogState,
    /// Memory update notification banner.
    pub memory_update_notification:
        crate::memory_update_notification::MemoryUpdateNotificationState,
    /// MCP elicitation dialog (form requested by an MCP server).
    pub elicitation: crate::elicitation_dialog::ElicitationDialogState,
    /// Model picker overlay (/model command).
    pub model_picker: ModelPickerState,
    /// Session browser overlay (/session, /resume, /rename, /export).
    pub session_browser: SessionBrowserState,
    /// Session branching overlay (Ctrl+B) — create and switch branches.
    pub session_branching: crate::session_branching::SessionBranchingState,
    /// Task progress overlay (Ctrl+T) — shows task status with toggle capability.
    pub tasks_overlay: TasksOverlay,
    /// Export format picker dialog (/export).
    pub export_dialog: ExportDialogState,
    /// Context window / rate limit visualization overlay (/context).
    pub context_viz: ContextVizState,
    /// MCP server approval dialog.
    pub mcp_approval: McpApprovalDialogState,
    /// Go to Line dialog (Ctrl+G in message pane).
    pub go_to_line_dialog: GoToLineDialog,
    /// Bypass-permissions startup confirmation dialog.
    /// Shown at startup when --dangerously-skip-permissions was passed.
    /// User must explicitly accept or the session exits.
    pub bypass_permissions_dialog: crate::bypass_permissions_dialog::BypassPermissionsDialogState,
    /// First-launch onboarding welcome dialog.
    pub onboarding_dialog: crate::onboarding_dialog::OnboardingDialogState,
    /// API key input dialog (opened from /connect for key-based providers).
    pub key_input_dialog: crate::key_input_dialog::KeyInputDialogState,
    /// Device code / browser auth dialog (GitHub Copilot device flow, Anthropic OAuth).
    pub device_auth_dialog: crate::device_auth_dialog::DeviceAuthDialogState,
    /// When set, the main loop should spawn the async auth task for this provider.
    pub device_auth_pending: Option<String>,
    /// Shared provider registry for dynamic model fetching.
    pub provider_registry: Option<std::sync::Arc<mangocode_api::ProviderRegistry>>,
    /// Model registry populated from models.dev — single source of truth for
    /// all provider models shown in the `/model` picker.
    pub model_registry: mangocode_api::ModelRegistry,
    /// When `true`, the main event loop should spawn an async task to fetch
    /// the model list from the current provider's `list_models()` API.
    pub model_picker_fetch_pending: bool,
    /// Credential store for provider API keys and OAuth tokens.
    pub auth_store: mangocode_core::AuthStore,
    /// Connect-a-provider dialog (/connect command).
    pub connect_dialog: DialogSelectState,
    /// Ctrl+K command palette overlay.
    pub command_palette: DialogSelectState,
    /// Slash-command metadata used by prompt typeahead and the command palette.
    pub prompt_slash_commands: Vec<SlashCommandSpec>,
    /// Whether MangoCode was launched from the user's home directory.
    /// Shown as a startup notice: "Note: You have launched claude in your home directory…"
    pub home_dir_warning: bool,
    /// Output style: "auto" | "stream" | "verbose".
    pub output_style: String,
    /// PR number for the current branch (None if not in a PR context).
    pub pr_number: Option<u32>,
    /// PR URL for the current branch.
    pub pr_url: Option<String>,
    /// PR review state: "approved", "changes_requested", "review_required", etc.
    pub pr_state: Option<String>,
    /// Count of in-progress background tasks (drives the footer pill).
    pub background_task_count: usize,
    /// Background task status text shown in footer pill.
    pub background_task_status: Option<String>,
    /// External status line command output (from CLAUDE_STATUS_COMMAND).
    pub status_line_override: Option<String>,
    /// Whether auto-compact is enabled (from settings).
    pub auto_compact_enabled: bool,
    /// Context threshold (0-100) at which to auto-compact.
    pub auto_compact_threshold: u8,

    // ---- Voice hold-to-talk ------------------------------------------------
    /// The global voice recorder, Some when voice is enabled in config.
    pub voice_recorder: Option<Arc<Mutex<mangocode_core::voice::VoiceRecorder>>>,
    /// True while recording is active (Alt+V toggled on).
    pub voice_recording: bool,
    /// Receiver for VoiceEvent messages produced by the recorder task.
    pub voice_event_rx: Option<tokio::sync::mpsc::Receiver<mangocode_core::voice::VoiceEvent>>,
    /// Receiver for model-list results fetched in the background when the
    /// /model picker opens.  Drained each frame so models appear as soon as
    /// the fetch completes.
    pub model_fetch_rx:
        Option<tokio::sync::mpsc::Receiver<Result<Vec<crate::model_picker::ModelEntry>, ()>>>,

    // ---- Context window & rate limit info ----------------------------------
    /// Total context window size for the current model (tokens).
    pub context_window_size: u64,
    /// How many tokens are currently used in the context window.
    pub context_used_tokens: u64,
    /// Rate limit info — 5-hour window usage percentage (0–100).
    pub rate_limit_5h_pct: Option<f32>,
    /// Rate limit info — 7-day window usage percentage (0–100).
    pub rate_limit_7day_pct: Option<f32>,
    /// Active worktree name (if in a worktree).
    pub worktree_name: Option<String>,
    /// Active worktree branch (if in a worktree).
    pub worktree_branch: Option<String>,
    /// Agent type badge: "agent" | "coordinator" | "subagent".
    pub agent_type_badge: Option<String>,

    // ---- Thinking block expansion state ----------------------------------
    /// Set of thinking block content hashes that are expanded.
    pub thinking_expanded: std::collections::HashSet<u64>,
    /// The message pane area from the last render frame (used for mouse hit testing).
    pub last_msg_area: Cell<ratatui::layout::Rect>,
    /// The frame region that supports text selection.
    pub last_selectable_area: Cell<ratatui::layout::Rect>,
    /// The prompt input area from the last render frame (used for focus routing).
    pub last_input_area: Cell<ratatui::layout::Rect>,
    /// Which area of the TUI currently has keyboard focus.
    pub focus: FocusTarget,
    /// Maps virtual_row_index → thinking_block_hash for click detection.
    pub thinking_row_map: RefCell<std::collections::HashMap<u16, u64>>,
    /// Maps screen row → transcript message index for right-click hit testing.
    pub message_row_map: RefCell<std::collections::HashMap<u16, usize>>,
    /// Hovered transcript message index from latest mouse-move event.
    pub hovered_message_index: Option<usize>,
    /// Total message lines from the last render (used for virtual row mapping).
    pub total_message_lines: Cell<usize>,
    /// Scroll offset from the last render frame (used for selection validation).
    pub last_render_scroll_offset: Cell<u16>,

    // ---- Text selection state --------------------------------------------
    /// Selection drag anchor (col, row) — set on mouse-down.
    pub selection_anchor: Option<(u16, u16)>,
    /// Selection drag focus (col, row) — updated on mouse-drag / mouse-up.
    pub selection_focus: Option<(u16, u16)>,
    /// Text extracted from the current selection (updated each render frame).
    pub selection_text: RefCell<String>,

    // ---- Advanced mouse interaction state --------------------------------
    /// Timestamp of the last left mouse click (for double/triple-click detection).
    pub last_click_time: Option<std::time::Instant>,
    /// Position of the last left mouse click (for double/triple-click detection).
    pub last_click_position: Option<(u16, u16)>,
    /// Count of consecutive clicks: 1 = single, 2 = double, 3+ = triple.
    pub click_count: u32,
    /// Context menu state: position and selected index.
    pub context_menu_state: Option<ContextMenuState>,

    // ---- Scroll acceleration state (trackpad feel) -----------------------
    /// Current acceleration multiplier for scroll events.
    scroll_accel: f32,
    /// Timestamp of the last scroll event (for burst detection).
    scroll_last_time: Option<std::time::Instant>,

    // ---- Bash prefix allowlist -------------------------------------------
    /// Command prefixes that have been permanently allowed this session via
    /// the "Allow commands starting with X" option in the bash permission dialog.
    /// Before showing the dialog for a bash command, the first whitespace-delimited
    /// word is checked against this set; a match silently auto-approves the request.
    pub bash_prefix_allowlist: std::collections::HashSet<String>,

    // ---- Auto-update notification ----------------------------------------
    /// If a newer version was found during background update check, this holds
    /// the latest version string (e.g. "0.1.0"). Shown in the footer status bar.
    pub update_available: Option<String>,

    // ---- Mascot image rendering -------------------------------------------
    /// Pre-encoded inline image escape sequence for the mango mascot
    /// (Kitty or Sixel). `None` if no inline image protocol is available.
    pub mascot_sixel: Option<String>,

    // ---- Paste burst guard -------------------------------------------------
    /// Number of printable characters received in a tight timing burst.
    pub paste_burst_printable_streak: usize,
    /// Timestamp of the last printable character used for burst detection.
    pub paste_burst_last_printable_at: Option<std::time::Instant>,
    /// If set and not expired, Enter inserts newline instead of submitting.
    pub paste_burst_until: Option<std::time::Instant>,
    /// Cursor position when the current paste burst started (for collapse).
    pub paste_burst_start_cursor: Option<usize>,
}

const SPINNER_VERBS: &[&str] = &[
    "Gooning",
    "Pegging",
    "Jestermaxxing",
    "Aurafarming",
    "Chumming",
    "Coreposting",
    "Vibecasting",
    "Dreamhoarding",
    "Schizoposting",
    "Deluluing",
    "Goblinizing",
    "Realitybending",
    "Sigmafying",
    "Rizzcrafting",
    "Moodboarding",
    "Dripfarming",
    "Rizzsummoning",
    "Enlightenmentgrifting",
    "NPCing",
    "Aestheticizing",
    "Mysticposting",
    "Flopdetecting",
    "Hyperfixating",
    "Brainrotating",
    "Vibechecking",
    "Selfgaslighting",
    "Frogbrooding",
    "Cloutweaving",
    "Overclocking",
    "Algorithmizing",
    "Maincharactering",
    "Simpsummoning",
    "Coreblending",
    "Chadsplaining",
    "Spiritbending",
    "Hyperlinking",
    "Realitymodding",
    "Dripmaxxing",
    "Uncannying",
    "Flopfishing",
    "Egocharging",
    "Goblincore-ing",
    "Bitrotting",
    "Wifihopping",
    "Schizotheorizing",
    "Neurallinking",
    "Situationshipping",
    "Moodglitching",
    "Rizzifying",
    "Lorebuilding",
    "Quantumvibing",
    "Chronomelding",
    "Backroomsdrifting",
    "Emojifying",
    "Sanitybuffering",
    "Clipcompiling",
    "Shadowbanning",
    "Rituallooping",
    "Overstimulating",
    "Fogposting",
    "Gremlinhopping",
    "Philosomemeing",
    "Aura-dripping",
    "Beefposting",
    "Enchantedpegging",
    "Vibequantizing",
    "Dripweaving",
    "Schizoascending",
    "Maxxing",
    "Overthinking",
    "Underachieving",
    "Dreamdissociating",
    "Goblinwalking",
    "Flopphasing",
    "Delulubranching",
    "Coremitosis",
    "Rizzlecasting",
    "Existentializing",
    "Auraamplifying",
    "Chungusing",
    "Quantumvibing",
    "Sleepparalyzing",
    "Vibe-overclocking",
    "Etherealizing",
    "Memecombobulating",
    "Moodrebounding",
    "Jankifying",
    "Liminalwalking",
    "Vibeosmosing",
    "Spiritbottling",
    "Psycheloading",
    "Finagling",
    "Basedifying",
    "Chronorotating",
    "Galaxymapping",
    "Astroprojecting",
    "Pegposting",
    "FrameMogging",
    "Vibe-huffing",
    "Ego-mining",
    "Delululooting",
    "Rizzrecharging",
    "Unrealizing",
    "Ejaculating",
    "Hyperlooping",
    "Mindmelting",
    "Auracombobulating",
    "Dreamsplitting",
    "Pegpivoting",
    "Squirting",
];

fn sample_spinner_verb(seed: usize) -> &'static str {
    SPINNER_VERBS[seed % SPINNER_VERBS.len()]
}

/// Past-tense verbs shown in the status row after a turn completes.
/// Mirrors `TURN_COMPLETION_VERBS` from `src/constants/turnCompletionVerbs.ts`.
const TURN_COMPLETION_VERBS: &[&str] = &[
    "Baked",
    "Brewed",
    "Churned",
    "Cogitated",
    "Cooked",
    "Crunched",
    "Pondered",
    "Processed",
    "Worked",
];

fn sample_completion_verb(seed: usize) -> &'static str {
    TURN_COMPLETION_VERBS[seed % TURN_COMPLETION_VERBS.len()]
}

/// Format a duration in seconds to a human-readable string, e.g. "2m 5s".
fn format_elapsed(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else {
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

impl App {
    pub fn new(config: Config, cost_tracker: Arc<CostTracker>) -> Self {
        let model_name = config.effective_model().to_string();
        let user_keybindings = UserKeybindings::load(&Settings::config_dir());
        let project_root = config
            .project_dir
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let prompt_slash_commands = prompt_slash_commands(&project_root, &config.skills);
        Self {
            config,
            cost_tracker,
            messages: Vec::new(),
            display_messages: Vec::new(),
            system_annotations: Vec::new(),
            input: String::new(),
            prompt_input: PromptInputState::new(),
            input_history: Vec::new(),
            history_index: None,
            scroll_offset: 0,
            is_streaming: false,
            streaming_text: String::new(),
            status_message: None,
            spinner_verb: None,
            should_quit: false,
            show_help: false,
            tool_use_blocks: Vec::new(),
            permission_request: None,
            frame_count: 0,
            token_count: 0,
            token_budget: Self::load_token_budget(),
            cost_usd: 0.0,
            model_name,
            has_credentials: true, // overridden by caller when no key is configured
            effort_level: EffortLevel::Normal,
            fast_mode: false,
            agent_mode: None,
            agent_status: Vec::new(),
            history_search: None,
            transcript_search: TranscriptSearchState::default(),
            keybindings: KeybindingResolver::new(&user_keybindings),
            cursor_pos: 0,
            auto_scroll: true,
            new_messages_while_scrolled: 0,
            token_warning_threshold_shown: 0,
            session_start: std::time::Instant::now(),
            turn_start: None,
            last_turn_elapsed: None,
            last_turn_verb: None,
            transcript_version: Cell::new(0),
            help_overlay: HelpOverlay::new(),
            history_search_overlay: HistorySearchOverlay::new(),
            global_search: GlobalSearchState::default(),
            message_selector: MessageSelectorOverlay::new(),
            rewind_flow: RewindFlowOverlay::new(),
            bridge_state: BridgeConnectionState::Disconnected,
            notifications: NotificationQueue::new(),
            plugin_hints: Vec::new(),
            session_title: None,
            remote_session_url: None,
            mcp_manager: None,
            pending_mcp_reconnect: false,
            file_history: None,
            current_turn: None,
            plan_mode: false,
            away_summary: None,
            stall_start: None,
            settings_screen: SettingsScreen::new(),
            theme_screen: ThemeScreen::new(),
            privacy_screen: PrivacyScreen::new(),
            stats_dialog: StatsDialogState::new(),
            mcp_view: McpViewState::new(),
            agents_menu: AgentsMenuState::new(),
            diff_viewer: DiffViewerState::new(),
            feedback_survey: crate::feedback_survey::FeedbackSurveyState::new(),
            memory_file_selector: crate::memory_file_selector::MemoryFileSelectorState::new(),
            hooks_config_menu: crate::hooks_config_menu::HooksConfigMenuState::new(),
            overage_upsell: crate::overage_upsell::OverageCreditUpsellState::new(),
            voice_mode_notice: crate::voice_mode_notice::VoiceModeNoticeState::new(),
            desktop_upsell: crate::desktop_upsell_startup::DesktopUpsellStartupState::new(),
            invalid_config_dialog: crate::invalid_config_dialog::InvalidConfigDialogState::new(),
            memory_update_notification:
                crate::memory_update_notification::MemoryUpdateNotificationState::new(),
            elicitation: crate::elicitation_dialog::ElicitationDialogState::new(),
            model_picker: ModelPickerState::new(),
            session_browser: SessionBrowserState::new(),
            session_branching: crate::session_branching::SessionBranchingState::new(),
            tasks_overlay: TasksOverlay::new(),
            export_dialog: ExportDialogState::new(),
            context_viz: ContextVizState::new(),
            mcp_approval: McpApprovalDialogState::new(),
            go_to_line_dialog: GoToLineDialog::new(),
            bypass_permissions_dialog:
                crate::bypass_permissions_dialog::BypassPermissionsDialogState::new(),
            onboarding_dialog: crate::onboarding_dialog::OnboardingDialogState::new(),
            key_input_dialog: crate::key_input_dialog::KeyInputDialogState::new(),
            device_auth_dialog: crate::device_auth_dialog::DeviceAuthDialogState::new(),
            device_auth_pending: None,
            provider_registry: None,
            model_registry: {
                let mut reg = mangocode_api::ModelRegistry::new();
                // Try to load cached models.dev data from disk.
                let cache_path = dirs::cache_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("mangocode")
                    .join("models.json");
                reg.load_cache(&cache_path);
                reg
            },
            model_picker_fetch_pending: false,
            auth_store: mangocode_core::AuthStore::load(),
            connect_dialog: {
                let items = vec![
                    // -- RECOMMENDED --
                    SelectItem {
                        id: "anthropic".into(),
                        title: "Anthropic (API Key)".into(),
                        description: "API key from console.anthropic.com".into(),
                        category: "Recommended".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "anthropic-max".into(),
                        title: "Claude Max (OAuth)".into(),
                        description: "Login with Claude Pro/Max subscription".into(),
                        category: "Recommended".into(),
                        badge: Some("OAUTH".into()),
                    },
                    SelectItem {
                        id: "openai".into(),
                        title: "OpenAI".into(),
                        description: "".into(),
                        category: "Recommended".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "google".into(),
                        title: "Google Gemini".into(),
                        description: "Direct Gemini API (AI Studio)".into(),
                        category: "Recommended".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "github-copilot".into(),
                        title: "GitHub Copilot".into(),
                        description: "".into(),
                        category: "Recommended".into(),
                        badge: None,
                    },
                    // -- FAST & FREE --
                    SelectItem {
                        id: "groq".into(),
                        title: "Groq".into(),
                        description: "".into(),
                        category: "Fast & Free".into(),
                        badge: Some("FREE".into()),
                    },
                    SelectItem {
                        id: "cerebras".into(),
                        title: "Cerebras".into(),
                        description: "".into(),
                        category: "Fast & Free".into(),
                        badge: Some("FREE".into()),
                    },
                    SelectItem {
                        id: "sambanova".into(),
                        title: "SambaNova".into(),
                        description: "".into(),
                        category: "Fast & Free".into(),
                        badge: Some("FREE".into()),
                    },
                    // -- LOCAL (no key needed) --
                    SelectItem {
                        id: "ollama".into(),
                        title: "Ollama".into(),
                        description: "Run models locally".into(),
                        category: "Local".into(),
                        badge: Some("LOCAL".into()),
                    },
                    SelectItem {
                        id: "lmstudio".into(),
                        title: "LM Studio".into(),
                        description: "Local model server".into(),
                        category: "Local".into(),
                        badge: Some("LOCAL".into()),
                    },
                    SelectItem {
                        id: "llamacpp".into(),
                        title: "llama.cpp".into(),
                        description: "C++ inference server".into(),
                        category: "Local".into(),
                        badge: Some("LOCAL".into()),
                    },
                    // -- AFFORDABLE --
                    SelectItem {
                        id: "deepseek".into(),
                        title: "DeepSeek".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "mistral".into(),
                        title: "Mistral".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "openrouter".into(),
                        title: "OpenRouter".into(),
                        description: "100+ models \u{00b7} One key".into(),
                        category: "Aggregator".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "togetherai".into(),
                        title: "Together AI".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "perplexity".into(),
                        title: "Perplexity".into(),
                        description: "Search-augmented AI".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "cohere".into(),
                        title: "Cohere".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "xai".into(),
                        title: "xAI".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "deepinfra".into(),
                        title: "DeepInfra".into(),
                        description: "".into(),
                        category: "Affordable".into(),
                        badge: None,
                    },
                    // -- ENTERPRISE --
                    SelectItem {
                        id: "azure".into(),
                        title: "Azure OpenAI".into(),
                        description: "".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "amazon-bedrock".into(),
                        title: "AWS Bedrock".into(),
                        description: "".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "google-vertex".into(),
                        title: "Google Vertex AI".into(),
                        description: "".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "google-vertex-access-token".into(),
                        title: "Google Vertex AI (Access Token)".into(),
                        description: "enter VERTEX_ACCESS_TOKEN".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "sap-ai-core".into(),
                        title: "SAP AI Core".into(),
                        description: "Enterprise AI platform".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "gitlab".into(),
                        title: "GitLab Duo".into(),
                        description: "AI in GitLab".into(),
                        category: "Enterprise".into(),
                        badge: None,
                    },
                    // -- CLOUD / GATEWAY --
                    SelectItem {
                        id: "cloudflare-ai-gateway".into(),
                        title: "Cloudflare AI Gateway".into(),
                        description: "".into(),
                        category: "Gateway".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "cloudflare-workers-ai".into(),
                        title: "Cloudflare Workers AI".into(),
                        description: "Edge AI inference".into(),
                        category: "Gateway".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "vercel".into(),
                        title: "Vercel AI Gateway".into(),
                        description: "AI SDK gateway".into(),
                        category: "Gateway".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "helicone".into(),
                        title: "Helicone".into(),
                        description: "AI observability gateway".into(),
                        category: "Gateway".into(),
                        badge: None,
                    },
                    // -- MORE PROVIDERS --
                    SelectItem {
                        id: "huggingface".into(),
                        title: "Hugging Face".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "nvidia".into(),
                        title: "Nvidia".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "alibaba".into(),
                        title: "".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "venice".into(),
                        title: "Venice AI".into(),
                        description: "Privacy-first AI".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "moonshotai".into(),
                        title: "".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "zhipuai".into(),
                        title: "".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "siliconflow".into(),
                        title: "SiliconFlow".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "nebius".into(),
                        title: "Nebius".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "novita".into(),
                        title: "Novita".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "minimax".into(),
                        title: "MiniMax".into(),
                        description: "Anthropic-compatible (M2.7)".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "ovhcloud".into(),
                        title: "OVHcloud".into(),
                        description: "EU-hosted AI".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "scaleway".into(),
                        title: "Scaleway".into(),
                        description: "EU cloud AI".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "vultr".into(),
                        title: "Vultr".into(),
                        description: "Cloud inference".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "baseten".into(),
                        title: "Baseten".into(),
                        description: "Model serving".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "friendli".into(),
                        title: "Friendli".into(),
                        description: "Serverless inference".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "upstage".into(),
                        title: "Upstage".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "stepfun".into(),
                        title: "StepFun".into(),
                        description: "".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                    SelectItem {
                        id: "fireworks".into(),
                        title: "Fireworks AI".into(),
                        description: "Fast inference".into(),
                        category: "More Providers".into(),
                        badge: None,
                    },
                ];
                DialogSelectState::new("Connect a Provider", items)
            },
            command_palette: {
                let items: Vec<SelectItem> = prompt_slash_commands
                    .iter()
                    .map(|cmd| SelectItem {
                        id: format!("/{}", cmd.name),
                        title: format!("/{}", cmd.name),
                        description: cmd.description.clone(),
                        category: cmd.group.clone(),
                        badge: None,
                    })
                    .collect();
                DialogSelectState::new("Command Palette", items)
            },
            prompt_slash_commands,
            home_dir_warning: false,
            output_style: "auto".to_string(),
            pr_number: None,
            pr_url: None,
            pr_state: None,
            background_task_count: 0,
            background_task_status: None,
            status_line_override: None,
            auto_compact_enabled: false,
            auto_compact_threshold: 95,
            voice_recorder: {
                // Check whether voice input has been enabled via the /voice command
                // (stored in ~/.mangocode/ui-settings.json).  We also accept
                // MANGOCODE_VOICE_ENABLED=1 as an override for easier testing.
                let voice_on = std::env::var("MANGOCODE_VOICE_ENABLED")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
                    || {
                        let path =
                            mangocode_core::config::Settings::config_dir().join("ui-settings.json");
                        std::fs::read_to_string(&path)
                            .ok()
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                            .and_then(|v| v["voice_enabled"].as_bool())
                            .unwrap_or(false)
                    };
                if voice_on {
                    let recorder = mangocode_core::voice::global_voice_recorder();
                    if let Ok(mut r) = recorder.lock() {
                        r.set_enabled(true);
                    }
                    Some(recorder)
                } else {
                    None
                }
            },
            voice_recording: false,
            voice_event_rx: None,
            model_fetch_rx: None,
            context_window_size: 0,
            context_used_tokens: 0,
            rate_limit_5h_pct: None,
            rate_limit_7day_pct: None,
            worktree_name: None,
            worktree_branch: None,
            agent_type_badge: None,
            thinking_expanded: std::collections::HashSet::new(),
            last_msg_area: Cell::new(ratatui::layout::Rect::default()),
            last_selectable_area: Cell::new(ratatui::layout::Rect::default()),
            last_input_area: Cell::new(ratatui::layout::Rect::default()),
            focus: FocusTarget::Input,
            thinking_row_map: RefCell::new(std::collections::HashMap::new()),
            message_row_map: RefCell::new(std::collections::HashMap::new()),
            hovered_message_index: None,
            total_message_lines: Cell::new(0),
            last_render_scroll_offset: Cell::new(0),
            selection_anchor: None,
            selection_focus: None,
            selection_text: RefCell::new(String::new()),
            last_click_time: None,
            last_click_position: None,
            click_count: 0,
            context_menu_state: None,
            scroll_accel: 3.0,
            scroll_last_time: None,
            bash_prefix_allowlist: std::collections::HashSet::new(),
            update_available: None,
            mascot_sixel: None,
            paste_burst_printable_streak: 0,
            paste_burst_last_printable_at: None,
            paste_burst_until: None,
            paste_burst_start_cursor: None,
        }
    }

    fn modal_stack(&self) -> Vec<ModalOwner> {
        let mut stack = Vec::new();

        macro_rules! push_if {
            ($cond:expr, $owner:expr) => {
                if $cond {
                    stack.push($owner);
                }
            };
        }

        push_if!(self.permission_request.is_some(), ModalOwner::PermissionRequest);
        push_if!(self.settings_screen.visible, ModalOwner::Settings);
        push_if!(self.theme_screen.visible, ModalOwner::Theme);
        push_if!(self.privacy_screen.visible, ModalOwner::Privacy);
        push_if!(self.stats_dialog.open, ModalOwner::Stats);
        push_if!(self.mcp_view.open, ModalOwner::McpView);
        push_if!(self.agents_menu.open, ModalOwner::AgentsMenu);
        push_if!(self.diff_viewer.open, ModalOwner::DiffViewer);
        push_if!(self.bypass_permissions_dialog.visible, ModalOwner::BypassPermissions);
        push_if!(self.onboarding_dialog.visible, ModalOwner::Onboarding);
        push_if!(self.device_auth_dialog.visible, ModalOwner::DeviceAuth);
        push_if!(self.key_input_dialog.visible, ModalOwner::KeyInput);
        push_if!(self.connect_dialog.visible, ModalOwner::Connect);
        push_if!(self.invalid_config_dialog.visible, ModalOwner::InvalidConfig);
        push_if!(self.desktop_upsell.visible, ModalOwner::DesktopUpsell);
        push_if!(
            self.memory_update_notification.visible,
            ModalOwner::MemoryUpdateNotification
        );
        push_if!(self.voice_mode_notice.visible, ModalOwner::VoiceModeNotice);
        push_if!(self.overage_upsell.visible, ModalOwner::OverageUpsell);
        push_if!(self.hooks_config_menu.visible, ModalOwner::HooksConfig);
        push_if!(self.memory_file_selector.visible, ModalOwner::MemoryFileSelector);
        push_if!(self.feedback_survey.visible, ModalOwner::FeedbackSurvey);
        push_if!(self.go_to_line_dialog.active, ModalOwner::GoToLineDialog);
        push_if!(self.mcp_approval.visible, ModalOwner::McpApproval);
        push_if!(self.context_viz.visible, ModalOwner::ContextViz);
        push_if!(self.export_dialog.visible, ModalOwner::ExportDialog);
        push_if!(self.session_branching.visible, ModalOwner::SessionBranching);
        push_if!(self.session_browser.visible, ModalOwner::SessionBrowser);
        push_if!(self.model_picker.visible, ModalOwner::ModelPicker);
        push_if!(self.elicitation.visible, ModalOwner::Elicitation);
        push_if!(self.command_palette.visible, ModalOwner::CommandPalette);
        push_if!(self.global_search.open, ModalOwner::GlobalSearch);
        push_if!(
            self.history_search_overlay.visible || self.history_search.is_some(),
            ModalOwner::HistorySearch
        );
        push_if!(self.help_overlay.visible || self.show_help, ModalOwner::Help);
        push_if!(self.tasks_overlay.visible, ModalOwner::Tasks);
        push_if!(self.rewind_flow.visible, ModalOwner::Rewind);

        stack
    }

    pub fn active_modal_owner(&self) -> Option<ModalOwner> {
        self.modal_stack().into_iter().next()
    }

    fn expire_paste_burst_if_idle(&mut self, now: std::time::Instant) {
        if self
            .paste_burst_until
            .is_some_and(|until| now >= until)
        {
            self.paste_burst_until = None;
            self.paste_burst_printable_streak = 0;
            self.paste_burst_last_printable_at = None;
            self.paste_burst_start_cursor = None;
        }
    }

    fn is_in_paste_burst(&self, now: std::time::Instant) -> bool {
        self.paste_burst_until
            .is_some_and(|until| now < until)
    }

    fn activate_paste_burst(&mut self, now: std::time::Instant) {
        self.paste_burst_until = Some(now + PASTE_BURST_IDLE_TIMEOUT);
    }

    fn register_printable_keystroke_for_paste_burst(
        &mut self,
        key: KeyEvent,
        now: std::time::Instant,
    ) {
        // Ignore control/meta chords from burst scoring.
        if key.modifiers.contains(KeyModifiers::CONTROL)
            || key.modifiers.contains(KeyModifiers::ALT)
            || key.modifiers.contains(KeyModifiers::SUPER)
        {
            return;
        }

        match self.paste_burst_last_printable_at {
            Some(prev) if now.duration_since(prev) <= PASTE_BURST_PRINTABLE_GAP => {
                self.paste_burst_printable_streak += 1;
            }
            _ => {
                self.paste_burst_printable_streak = 1;
                // New burst — record where the prompt cursor is now
                self.paste_burst_start_cursor = Some(self.prompt_input.cursor);
            }
        }
        self.paste_burst_last_printable_at = Some(now);

        if self.paste_burst_printable_streak >= PASTE_BURST_MIN_STREAK {
            self.activate_paste_burst(now);

            // If we've accumulated a lot of rapid input, collapse it into a
            // neutral placeholder token while preserving full pasted content.
            if let Some(start) = self.paste_burst_start_cursor {
                let burst_len = self.prompt_input.cursor.saturating_sub(start);
                if burst_len > 1024 {
                    // Extract the burst text, collapse it
                    let burst_text: String = self.prompt_input.text[start..self.prompt_input.cursor].to_string();
                    let line_count = burst_text.lines().count();
                    self.prompt_input.paste_counter += 1;
                    let placeholder = if line_count > 1 {
                        format!("[Blob:{} lines={}]", self.prompt_input.paste_counter, line_count)
                    } else {
                        format!("[Blob:{}]", self.prompt_input.paste_counter)
                    };
                    // Store the original content for when the message is submitted
                    self.prompt_input.paste_contents.insert(
                        self.prompt_input.paste_counter,
                        burst_text,
                    );
                    // Replace the burst text with the placeholder
                    self.prompt_input.text.replace_range(start..self.prompt_input.cursor, &placeholder);
                    self.prompt_input.cursor = start + placeholder.len();
                    self.prompt_input.update_token_estimate();
                    // Reset burst tracking for the next paste
                    self.paste_burst_start_cursor = None;
                    self.paste_burst_printable_streak = 0;
                }
            }
        }
    }

    fn handle_prompt_paste(&mut self, data: &str, now: std::time::Instant) {
        self.prompt_input.paste(data);
        self.refresh_prompt_input();
        if data.contains('\n') || data.contains('\r') {
            self.activate_paste_burst(now);
        }
    }

    /// Load token budget from environment or model defaults.
    /// Returns Some(max_tokens) if available, None otherwise.
    /// Only enabled when the `token_budget` feature flag is active.
    #[cfg(feature = "token_budget")]
    fn load_token_budget() -> Option<u32> {
        // First check MANGOCODE_TOKEN_BUDGET env var
        if let Ok(budget_str) = std::env::var("MANGOCODE_TOKEN_BUDGET") {
            if let Ok(budget) = budget_str.parse::<u32>() {
                return Some(budget);
            }
        }
        // Could extend this to check model defaults, but for now just env var
        None
    }

    #[cfg(not(feature = "token_budget"))]
    fn load_token_budget() -> Option<u32> {
        None
    }

    fn display_default_model_for_provider(&self, provider_id: &str) -> String {
        if let Some(best) = self.model_registry.best_model_for_provider(provider_id) {
            if provider_id == "anthropic" {
                best
            } else {
                format!("{}/{}", provider_id, best)
            }
        } else {
            crate::model_picker::default_model_for_provider(provider_id)
        }
    }

    fn persist_provider_and_model(&self) {
        let mut settings = Settings::load_sync().unwrap_or_default();
        settings.provider = self.config.provider.clone();
        settings.config.provider = self.config.provider.clone();
        settings.config.model = self.config.model.clone();
        let _ = settings.save_sync();
    }

    fn infer_provider_from_model(model: &str) -> Option<String> {
        if let Some((provider, _)) = model.split_once('/') {
            let known = [
                "anthropic",
                "openai",
                "google",
                "groq",
                "cerebras",
                "deepseek",
                "mistral",
                "xai",
                "openrouter",
                "github-copilot",
                "cohere",
                "perplexity",
                "togetherai",
                "together-ai",
                "deepinfra",
                "venice",
                "minimax",
                "ollama",
                "lmstudio",
                "llamacpp",
                "azure",
                "amazon-bedrock",
            ];
            if known.contains(&provider) {
                return Some(provider.to_string());
            }
        }

        if model.starts_with("claude") {
            Some("anthropic".to_string())
        } else if model.starts_with("gpt-")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4")
        {
            Some("openai".to_string())
        } else if model.to_ascii_lowercase().starts_with("minimax") {
            Some("minimax".to_string())
        } else if model.starts_with("gemini") || model.starts_with("gemma") {
            Some("google".to_string())
        } else {
            None
        }
    }

    /// Switch the active provider while clearing any explicit model override.
    fn set_provider_default(&mut self, provider_id: String) {
        self.config.provider = Some(provider_id.clone());
        self.config.model = None;

        let model = self.display_default_model_for_provider(&provider_id);
        self.cost_tracker.set_model(&model);
        self.model_name = model;
    }

    /// Update the active model name (also updates config + cost tracker).
    pub fn set_model(&mut self, model: String) {
        self.cost_tracker.set_model(&model);
        self.model_name = model.clone();
        self.config.model = Some(model.clone());
        if let Some(provider) = Self::infer_provider_from_model(&model) {
            self.config.provider = Some(provider);
        }
    }

    /// Apply a theme by name, persisting it to config.
    pub fn apply_theme(&mut self, theme_name: &str) {
        let theme = match theme_name {
            "dark" => Theme::Dark,
            "light" => Theme::Light,
            "default" => Theme::Default,
            "deuteranopia" => Theme::Deuteranopia,
            other => Theme::Custom(other.to_string()),
        };
        self.config.theme = theme;
        // Persist to settings file
        let mut settings = Settings::load_sync().unwrap_or_default();
        settings.config.theme = self.config.theme.clone();
        let _ = settings.save_sync();
        self.status_message = Some(format!("Theme set to: {}", theme_name));
    }

    /// Handle slash commands that should open UI screens rather than execute
    /// as normal commands. Returns `true` if the command was intercepted.
    pub fn intercept_slash_command(&mut self, cmd: &str) -> bool {
        self.close_secondary_views();
        match cmd {
            "config" | "settings" => {
                self.settings_screen.open();
                true
            }
            "theme" => {
                let current = match &self.config.theme {
                    Theme::Dark => "dark",
                    Theme::Light => "light",
                    Theme::Default => "default",
                    Theme::Deuteranopia => "deuteranopia",
                    Theme::Custom(s) => s.as_str(),
                };
                self.theme_screen.open(current);
                true
            }
            "privacy-settings" | "privacy" => {
                self.privacy_screen.open();
                true
            }
            "stats" => {
                self.stats_dialog.open();
                true
            }
            "mcp" => {
                let servers = self.load_mcp_servers();
                self.mcp_view.open(servers);
                true
            }
            "agents" => {
                self.open_agents_menu();
                true
            }
            "diff" | "review" => {
                let root = self.project_root();
                self.diff_viewer.open(&root);
                true
            }
            "changes" => {
                let root = self.project_root();
                self.refresh_turn_diff_from_history();
                self.diff_viewer.open_turn(&root);
                true
            }
            "search" | "find" => {
                self.global_search.open();
                true
            }
            "survey" | "feedback" => {
                self.feedback_survey.open();
                true
            }
            "memory" => {
                let root = self.project_root();
                self.memory_file_selector.open(&root);
                true
            }
            "hooks" => {
                self.hooks_config_menu.open();
                true
            }
            "connect" => {
                self.connect_dialog.open();
                true
            }
            "model" => {
                let provider = self.config.provider.as_deref().unwrap_or("anthropic");

                // Reload cache from disk in case the background models.dev fetch
                // has written new data since we last opened the picker.
                let cache_path = dirs::cache_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("mangocode")
                    .join("models.json");
                if cache_path.exists() {
                    self.model_registry.load_cache(&cache_path);
                }

                // Get models from the registry (models.dev data); falls back to
                // hardcoded when the registry has no data for this provider.
                let models = crate::model_picker::models_for_provider_from_registry(
                    provider,
                    &self.model_registry,
                );
                self.model_picker.set_models(models);
                self.model_picker_fetch_pending = true;

                let provider_prefix = format!("{}/", provider);
                let current = self
                    .model_name
                    .strip_prefix(&provider_prefix)
                    .unwrap_or(self.model_name.as_str())
                    .to_string();
                let effort = self.effort_level;
                let fast = self.fast_mode;
                self.model_picker.open_with_state(&current, effort, fast);

                true
            }
            "session" | "resume" => {
                self.session_browser.open(vec![]);
                true
            }
            "clear" => {
                self.messages.clear();
                self.system_annotations.clear();
                self.display_messages.clear();
                self.streaming_text.clear();
                self.tool_use_blocks.clear();
                self.invalidate_transcript();
                self.status_message = Some("Conversation cleared.".to_string());
                true
            }
            "exit" | "quit" => {
                self.should_quit = true;
                true
            }
            "vim" => {
                self.prompt_input.vim_enabled = !self.prompt_input.vim_enabled;
                let status = if self.prompt_input.vim_enabled {
                    "enabled"
                } else {
                    "disabled"
                };
                self.status_message = Some(format!("Vim mode {}.", status));
                self.refresh_prompt_input();
                true
            }
            "fast" => {
                self.fast_mode = !self.fast_mode;
                let status = if self.fast_mode {
                    "enabled"
                } else {
                    "disabled"
                };
                self.status_message = Some(format!("Fast mode {}.", status));
                true
            }
            "plan" => {
                use mangocode_core::config::PermissionMode;
                self.plan_mode = !self.plan_mode;
                self.config.permission_mode = if self.plan_mode {
                    PermissionMode::Plan
                } else {
                    PermissionMode::Default
                };
                self.status_message = Some(if self.plan_mode {
                    "Plan mode ON — MangoCode will plan before acting.".to_string()
                } else {
                    "Plan mode OFF.".to_string()
                });
                // Allow CLI path to also run (sends UserMessage to MangoCode).
                false
            }
            "compact" => {
                // Handled by execute_command in the CLI loop (real LLM compaction).
                false
            }
            "copy" => {
                // Copy last assistant message to clipboard. Attempt arboard; fall back to notification.
                let last = self
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == Role::Assistant)
                    .map(|m| m.get_all_text());
                if let Some(text) = last {
                    // Try xclip/xsel/pbcopy/clip.exe for clipboard; fall back to notification.
                    let copied = try_copy_to_clipboard(&text);
                    if copied {
                        self.notifications.push(
                            NotificationKind::Info,
                            "Copied to clipboard.".to_string(),
                            Some(3),
                        );
                    } else {
                        self.notifications.push(
                            NotificationKind::Info,
                            format!(
                                "Last response: {} chars (clipboard unavailable)",
                                text.len()
                            ),
                            Some(5),
                        );
                    }
                } else {
                    self.notifications.push(
                        NotificationKind::Warning,
                        "No assistant message to copy.".to_string(),
                        Some(3),
                    );
                }
                true
            }
            "output-style" => {
                self.output_style = match self.output_style.as_str() {
                    "auto" => "stream".to_string(),
                    "stream" => "verbose".to_string(),
                    _ => "auto".to_string(),
                };
                self.status_message = Some(format!("Output style: {}.", self.output_style));
                true
            }
            "effort" => {
                // Only cycle the visual indicator when called with no args (arg-based
                // effort changes are handled by execute_command + main.rs sync).
                self.effort_level = match self.effort_level {
                    EffortLevel::Low => EffortLevel::Normal,
                    EffortLevel::Normal => EffortLevel::High,
                    EffortLevel::High => EffortLevel::Max,
                    EffortLevel::Max => EffortLevel::Low,
                };
                self.status_message = Some(format!(
                    "Effort: {} {}",
                    self.effort_level.symbol(),
                    self.effort_level.label(),
                ));
                true
            }
            "voice" => {
                let was_on = self.voice_recorder.is_some();
                if was_on {
                    self.voice_recorder = None;
                    self.voice_mode_notice.dismiss();
                    self.status_message = Some("Voice mode disabled.".to_string());
                } else {
                    let recorder = mangocode_core::voice::global_voice_recorder();
                    if let Ok(mut r) = recorder.lock() {
                        r.set_enabled(true);
                    }
                    self.voice_recorder = Some(recorder);
                    self.voice_mode_notice = crate::voice_mode_notice::VoiceModeNoticeState::new();
                    self.status_message =
                        Some("Voice mode enabled. Press Alt+V to record.".to_string());
                }
                true
            }
            "doctor" => {
                // Handled by execute_command (DoctorCommand).
                false
            }
            "cost" => {
                self.stats_dialog.open();
                true
            }
            "rewind" => {
                self.open_rewind_flow();
                true
            }
            "export" => {
                self.export_dialog.open();
                true
            }
            "context" => {
                self.context_viz.toggle();
                true
            }
            "rename" => {
                self.session_browser.open(vec![]);
                self.session_browser.start_rename();
                true
            }
            "init" | "login" | "logout" => {
                // Handled by execute_command (CLI-level operations).
                false
            }
            "keybindings" => {
                // Open settings on KeyBindings tab
                self.settings_screen.open();
                self.settings_screen.active_tab = crate::settings_screen::SettingsTab::KeyBindings;
                true
            }
            "help" => {
                // Open the help overlay (same as pressing F1).
                if !self.help_overlay.visible {
                    self.show_help = true;
                    self.help_overlay.toggle();
                }
                true
            }
            _ => false,
        }
    }

    fn close_secondary_views(&mut self) {
        self.stats_dialog.close();
        self.mcp_view.close();
        self.agents_menu.close();
        self.diff_viewer.close();
        self.feedback_survey.close();
        self.memory_file_selector.close();
        self.hooks_config_menu.close();
        self.model_picker.close();
        self.session_browser.close();
        self.session_branching.close();
        self.tasks_overlay.close();
        self.export_dialog.dismiss();
        self.context_viz.close();
        self.connect_dialog.close();
        self.command_palette.close();
        self.key_input_dialog.close();
        self.device_auth_dialog.close();
        self.settings_screen.close();
        self.theme_screen.close();
        self.privacy_screen.close();
    }

    /// Perform the export based on the selected format. Returns the path written.
    pub fn perform_export(&mut self) -> Option<String> {
        use crate::export_dialog::{export_as_json, export_as_markdown};
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let (filename, content) = match self.export_dialog.selected {
            ExportFormat::Json => {
                let json = export_as_json(&self.messages, self.session_title.as_deref());
                let s = serde_json::to_string_pretty(&json).unwrap_or_default();
                (format!("claude-export-{}.json", ts), s)
            }
            ExportFormat::Markdown => {
                let md = export_as_markdown(&self.messages, self.session_title.as_deref());
                (format!("claude-export-{}.md", ts), md)
            }
        };
        if std::fs::write(&filename, &content).is_ok() {
            self.export_dialog.dismiss();
            Some(filename)
        } else {
            None
        }
    }

    fn project_root(&self) -> std::path::PathBuf {
        self.config
            .project_dir
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    fn refresh_global_search(&mut self) {
        let root = self.project_root();
        self.global_search.run_search(&root);
        self.sync_transcript_search_from_global();
    }

    fn transcript_search_haystack(&self) -> String {
        let mut parts: Vec<String> = self.messages.iter().map(Message::get_all_text).collect();
        parts.extend(self.system_annotations.iter().map(|ann| ann.text.clone()));
        parts.join("\n")
    }

    fn count_case_insensitive_occurrences(haystack: &str, needle: &str) -> usize {
        if needle.is_empty() {
            return 0;
        }
        let hay = haystack.to_lowercase();
        let need = needle.to_lowercase();
        let mut count = 0usize;
        let mut start = 0usize;
        while let Some(pos) = hay[start..].find(&need) {
            count += 1;
            start += pos + need.len();
        }
        count
    }

    pub(crate) fn recompute_transcript_search(&mut self) {
        if self.transcript_search.query.is_empty() {
            self.transcript_search.total_matches = 0;
            self.transcript_search.current_match = None;
            return;
        }

        self.invalidate_transcript();
        let haystack = self.transcript_search_haystack();
        let total = Self::count_case_insensitive_occurrences(&haystack, &self.transcript_search.query);
        self.transcript_search.total_matches = total;
        self.transcript_search.current_match = if total == 0 {
            None
        } else {
            Some(self.transcript_search.current_match.unwrap_or(0).min(total - 1))
        };
    }

    fn sync_transcript_search_from_global(&mut self) {
        let next_query = self.global_search.query.trim().to_string();
        if self.transcript_search.query != next_query {
            self.transcript_search.query = next_query;
            self.transcript_search.current_match = if self.transcript_search.query.is_empty() {
                None
            } else {
                Some(0)
            };
        }
        self.recompute_transcript_search();
    }

    fn transcript_search_step(&mut self, next: bool) {
        self.recompute_transcript_search();
        if self.transcript_search.total_matches == 0 {
            self.status_message = Some("Find: 0/0".to_string());
            return;
        }

        let total = self.transcript_search.total_matches;
        let cur = self.transcript_search.current_match.unwrap_or(0);
        let new_cur = if next {
            (cur + 1) % total
        } else if cur == 0 {
            total - 1
        } else {
            cur - 1
        };
        self.transcript_search.current_match = Some(new_cur);
        self.status_message = Some(format!("Find: {}/{}", new_cur + 1, total));
    }

    fn load_mcp_servers(&self) -> Vec<McpServerView> {
        if let Some(manager) = self.mcp_manager.as_ref() {
            let tool_defs = manager.all_tool_definitions();
            return self
                .config
                .mcp_servers
                .iter()
                .map(|server| {
                    let transport = server
                        .url
                        .as_ref()
                        .map(|_| server.server_type.clone())
                        .or_else(|| server.command.as_ref().map(|_| "stdio".to_string()))
                        .unwrap_or_else(|| server.server_type.clone());

                    let tools: Vec<McpToolView> = tool_defs
                        .iter()
                        .filter(|(server_name, _)| server_name == &server.name)
                        .map(|(_, tool_def)| McpToolView {
                            name: tool_def
                                .name
                                .strip_prefix(&format!("{}_", server.name))
                                .unwrap_or(&tool_def.name)
                                .to_string(),
                            server: server.name.clone(),
                            description: tool_def.description.clone(),
                            input_schema: Some(tool_def.input_schema.to_string()),
                        })
                        .collect();

                    let (status, error_message) = match manager.server_status(&server.name) {
                        mangocode_mcp::McpServerStatus::Connected { .. } => {
                            (McpViewStatus::Connected, None)
                        }
                        mangocode_mcp::McpServerStatus::Connecting => {
                            (McpViewStatus::Connecting, None)
                        }
                        mangocode_mcp::McpServerStatus::Disconnected { last_error } => {
                            if last_error.is_some() {
                                (McpViewStatus::Error, last_error)
                            } else {
                                (McpViewStatus::Disconnected, None)
                            }
                        }
                        mangocode_mcp::McpServerStatus::Failed { error, .. } => {
                            (McpViewStatus::Error, Some(error))
                        }
                    };

                    let catalog = manager.server_catalog(&server.name);
                    McpServerView {
                        name: server.name.clone(),
                        transport,
                        status,
                        tool_count: catalog
                            .as_ref()
                            .map(|entry| entry.tool_count)
                            .unwrap_or_else(|| tools.len()),
                        resource_count: catalog
                            .as_ref()
                            .map(|entry| entry.resource_count)
                            .unwrap_or(0),
                        prompt_count: catalog
                            .as_ref()
                            .map(|entry| entry.prompt_count)
                            .unwrap_or(0),
                        resources: catalog
                            .as_ref()
                            .map(|entry| entry.resources.clone())
                            .unwrap_or_default(),
                        prompts: catalog
                            .as_ref()
                            .map(|entry| entry.prompts.clone())
                            .unwrap_or_default(),
                        error_message,
                        tools,
                    }
                })
                .collect();
        }

        self.config
            .mcp_servers
            .iter()
            .map(|server| {
                let transport = server
                    .url
                    .as_ref()
                    .map(|_| server.server_type.clone())
                    .or_else(|| server.command.as_ref().map(|_| "stdio".to_string()))
                    .unwrap_or_else(|| server.server_type.clone());
                let description = if let Some(url) = &server.url {
                    format!("Endpoint: {}", url)
                } else if let Some(command) = &server.command {
                    let args = if server.args.is_empty() {
                        String::new()
                    } else {
                        format!(" {}", server.args.join(" "))
                    };
                    format!("Command: {}{}", command, args)
                } else {
                    "Configured server".to_string()
                };
                McpServerView {
                    name: server.name.clone(),
                    transport,
                    status: McpViewStatus::Disconnected,
                    tool_count: 0,
                    resource_count: 0,
                    prompt_count: 0,
                    resources: vec![],
                    prompts: vec![],
                    error_message: None,
                    tools: vec![McpToolView {
                        name: "connection".to_string(),
                        server: server.name.clone(),
                        description,
                        input_schema: None,
                    }],
                }
            })
            .collect()
    }

    fn open_agents_menu(&mut self) {
        let root = self.project_root();
        self.agents_menu.open(&root);
        self.agents_menu.active_agents = self
            .agent_status
            .iter()
            .enumerate()
            .map(|(idx, (name, status))| AgentInfo {
                id: format!("agent-{}", idx + 1),
                name: name.clone(),
                status: match status.as_str() {
                    "running" => AgentStatus::Running,
                    "waiting" | "waiting_for_tool" => AgentStatus::WaitingForTool,
                    "complete" | "completed" | "done" => AgentStatus::Complete,
                    "failed" | "error" => AgentStatus::Failed,
                    _ => AgentStatus::Idle,
                },
                current_tool: None,
                turns_completed: 0,
                is_coordinator: false,
                last_output: Some(status.clone()),
            })
            .collect();
    }

    /// Add a message directly (e.g. from a non-streaming source).
    pub fn add_message(&mut self, role: Role, text: String) {
        let msg = match role {
            Role::User => Message::user(text),
            Role::Assistant => Message::assistant(text),
        };
        if matches!(role, Role::User) {
            // Start latency timing at prompt submission time.
            self.turn_start = Some(std::time::Instant::now());
            self.last_turn_elapsed = None;
            self.last_turn_verb = None;
        }
        self.messages.push(msg);
        self.invalidate_transcript();
        self.on_new_message();
        self.recompute_transcript_search();
    }

    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
        self.invalidate_transcript();
        self.recompute_transcript_search();
    }

    pub fn push_message(&mut self, message: Message) {
        if matches!(message.role, Role::User) {
            // Start latency timing at prompt submission time.
            self.turn_start = Some(std::time::Instant::now());
            self.last_turn_elapsed = None;
            self.last_turn_verb = None;
        }
        self.messages.push(message);
        self.invalidate_transcript();
        self.on_new_message();
        self.recompute_transcript_search();
    }

    /// Push a synthetic system annotation into the conversation pane.
    /// It will appear after the current last message.
    pub fn push_system_message(&mut self, text: String, style: SystemMessageStyle) {
        self.system_annotations.push(SystemAnnotation {
            after_index: self.messages.len(),
            text,
            style,
        });
        self.invalidate_transcript();
    }

    /// Called whenever a new message is appended to `messages`.
    /// Manages the auto-scroll / new-message-counter state.
    fn on_new_message(&mut self) {
        if self.auto_scroll {
            // Auto-scroll: keep offset at 0 so render shows the bottom.
            self.scroll_offset = 0;
        } else {
            self.new_messages_while_scrolled = self.new_messages_while_scrolled.saturating_add(1);
        }
    }

    pub fn invalidate_transcript(&self) {
        self.transcript_version
            .set(self.transcript_version.get().wrapping_add(1));
    }

    /// Check current token usage and push token warning notifications as
    /// appropriate.  Call this after updating `token_count`.
    pub fn check_token_warnings(&mut self) {
        let window = mangocode_query::context_window_for_model(&self.model_name) as u32;
        if window == 0 {
            return;
        }
        let pct = (self.token_count as f64 / window as f64 * 100.0) as u8;

        // Only escalate — never repeat a threshold already shown.
        if pct >= 100 && self.token_warning_threshold_shown < 100 {
            self.token_warning_threshold_shown = 100;
            self.notifications.push(
                NotificationKind::Error,
                "Context window full. Running auto-compact\u{2026}".to_string(),
                None,
            );
        } else if pct >= 95 && self.token_warning_threshold_shown < 95 {
            self.token_warning_threshold_shown = 95;
            self.notifications.push(
                NotificationKind::Error,
                "Context window 95% full! Run /compact now.".to_string(),
                None, // persistent until dismissed
            );
        } else if pct >= 80 && self.token_warning_threshold_shown < 80 {
            self.token_warning_threshold_shown = 80;
            self.notifications.push(
                NotificationKind::Warning,
                "Context window 80% full. Consider /compact.".to_string(),
                Some(30),
            );
        }
    }

    /// Take the current input buffer, push it to history, and return it.
    pub fn take_input(&mut self) -> String {
        let input = self.prompt_input.take_expanded();
        if !input.is_empty() {
            self.prompt_input.history.push(input.clone());
            self.prompt_input.history_pos = None;
            self.prompt_input.history_draft.clear();
            self.input_history = self.prompt_input.history.clone();
            self.history_index = self.prompt_input.history_pos;
        }
        self.refresh_prompt_input();
        input
    }

    /// Compute the number of lines to scroll per wheel/trackpad event.
    /// Implements a simple acceleration model: rapid events (< 40 ms apart) are
    /// treated as trackpad bursts and accelerate up to 2×; slower events (mouse
    /// wheel) stay at the base 3-line step.
    fn scroll_step(&mut self) -> usize {
        let now = std::time::Instant::now();
        let elapsed_ms = self
            .scroll_last_time
            .map(|t| now.duration_since(t).as_millis())
            .unwrap_or(u128::MAX);
        self.scroll_last_time = Some(now);
        if elapsed_ms < 40 {
            // Trackpad burst — gradually accelerate
            self.scroll_accel = (self.scroll_accel + 0.4).min(6.0);
        } else {
            // Mouse click or first event — reset to base
            self.scroll_accel = 3.0;
        }
        self.scroll_accel.round() as usize
    }

    /// Open the rewind flow with the current message list converted to
    /// `SelectorMessage` entries.
    pub fn open_rewind_flow(&mut self) {
        let selector_msgs: Vec<SelectorMessage> = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let text = m.get_all_text();
                let preview: String = text.chars().take(80).collect();
                let has_tool_use = !m.get_tool_use_blocks().is_empty();
                SelectorMessage {
                    idx: i,
                    role: format!("{:?}", m.role).to_lowercase(),
                    preview,
                    has_tool_use,
                }
            })
            .collect();
        self.rewind_flow.open(selector_msgs);
    }

    /// Return the elapsed session time as a human-readable string, e.g. "2m 5s".
    pub fn elapsed_str(&self) -> String {
        let secs = self.session_start.elapsed().as_secs();
        if secs < 60 {
            format!("{}s", secs)
        } else {
            format!("{}m {}s", secs / 60, secs % 60)
        }
    }

    fn prompt_mode(&self) -> InputMode {
        if self.is_streaming {
            InputMode::Readonly
        } else if self.plan_mode {
            InputMode::Plan
        } else {
            InputMode::Default
        }
    }

    fn sync_legacy_prompt_fields(&mut self) {
        self.input = self.prompt_input.text.clone();
        self.cursor_pos = self.prompt_input.cursor;
        self.history_index = self.prompt_input.history_pos;
    }

    fn refresh_prompt_input(&mut self) {
        self.prompt_input.mode = self.prompt_mode();
        self.prompt_input.update_suggestions(&self.prompt_slash_commands);
        self.sync_legacy_prompt_fields();
    }

    pub fn set_prompt_text(&mut self, text: String) {
        self.prompt_input.replace_text(text);
        self.refresh_prompt_input();
    }

    // -----------------------------------------------------------------------
    // Voice PTT helpers
    // -----------------------------------------------------------------------

    /// Start PTT recording: open the microphone capture stream and signal the
    /// UI.  No-op when no voice recorder is attached or recording is already
    /// in progress.
    pub fn handle_voice_ptt_start(&mut self) {
        if self.voice_recording || self.voice_recorder.is_none() {
            return;
        }
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        self.voice_event_rx = Some(rx);
        self.voice_recording = true;
        if let Some(ref recorder_arc) = self.voice_recorder {
            let recorder = recorder_arc.clone();
            tokio::task::spawn_blocking(move || {
                if let Ok(mut r) = recorder.lock() {
                    tokio::runtime::Handle::current()
                        .block_on(r.start_recording(tx))
                        .ok();
                }
            });
        }
        self.status_message =
            Some("Recording\u{2026} release V or press Enter to transcribe".to_string());
    }

    /// Stop PTT recording: flip the AtomicBool inside VoiceRecorder so the
    /// capture thread exits, then fire a "Transcribing…" notice.  The
    /// transcript text arrives later via `voice_event_rx` and is injected into
    /// the prompt by the event-loop drain.
    pub fn handle_voice_ptt_stop(&mut self) {
        if !self.voice_recording {
            return;
        }
        self.voice_recording = false;
        if let Some(ref recorder_arc) = self.voice_recorder {
            let recorder = recorder_arc.clone();
            tokio::task::spawn_blocking(move || {
                if let Ok(mut r) = recorder.lock() {
                    tokio::runtime::Handle::current()
                        .block_on(r.stop_recording())
                        .ok();
                }
            });
        }
        self.status_message = Some("Transcribing\u{2026}".to_string());
    }

    pub fn attach_turn_diff_state(
        &mut self,
        file_history: Arc<parking_lot::Mutex<FileHistory>>,
        current_turn: Arc<std::sync::atomic::AtomicUsize>,
    ) {
        self.file_history = Some(file_history);
        self.current_turn = Some(current_turn);
        self.refresh_turn_diff_from_history();
    }

    pub fn attach_mcp_manager(&mut self, mcp_manager: Arc<mangocode_mcp::McpManager>) {
        self.mcp_manager = Some(mcp_manager);
    }

    pub fn refresh_mcp_view(&mut self) {
        let servers = self.load_mcp_servers();
        self.mcp_view.open(servers);
    }

    pub fn take_pending_mcp_reconnect(&mut self) -> bool {
        let pending = self.pending_mcp_reconnect;
        self.pending_mcp_reconnect = false;
        pending
    }

    /// Returns and clears any pending MCP approval result.
    pub fn take_mcp_approval_result(&mut self) -> Option<crate::dialogs::McpApprovalChoice> {
        if !self.mcp_approval.visible {
            return None;
        }
        // The dialog closes itself on confirm; we check if it's now closed
        None // Actual result is read by CLI loop via mcp_approval.visible + confirm()
    }

    /// Detect the current PR from environment variables or git.
    pub fn detect_pr(&mut self) {
        // Check CLAUDE_PR_NUMBER and CLAUDE_PR_URL env vars
        if let Ok(num) = std::env::var("CLAUDE_PR_NUMBER") {
            if let Ok(n) = num.parse::<u32>() {
                self.pr_number = Some(n);
            }
        }
        if let Ok(url) = std::env::var("CLAUDE_PR_URL") {
            self.pr_url = Some(url);
        }
        if let Ok(state) = std::env::var("CLAUDE_PR_STATE") {
            if !state.trim().is_empty() {
                self.pr_state = Some(state.trim().to_string());
            }
        }
        // Fall back to gh CLI if no env vars
        if self.pr_number.is_none() {
            if let Ok(output) = std::process::Command::new("gh")
                .args(["pr", "view", "--json", "number,url", "--jq", ".number,.url"])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = text.trim().split('\n').collect();
                    if parts.len() >= 2 {
                        if let Ok(n) = parts[0].trim().parse::<u32>() {
                            self.pr_number = Some(n);
                            self.pr_url = Some(parts[1].trim().to_string());
                        }
                    }
                }
            }
        }
    }

    fn clear_prompt(&mut self) {
        self.prompt_input.clear();
        self.refresh_prompt_input();
    }

    fn refresh_turn_diff_from_history(&mut self) {
        let Some(file_history) = self.file_history.as_ref() else {
            self.diff_viewer.set_turn_diff(Vec::new());
            return;
        };
        let Some(current_turn) = self.current_turn.as_ref() else {
            self.diff_viewer.set_turn_diff(Vec::new());
            return;
        };

        let turn_index = current_turn.load(std::sync::atomic::Ordering::Relaxed);
        if turn_index == 0 {
            self.diff_viewer.set_turn_diff(Vec::new());
            return;
        }

        let root = self.project_root();
        let files = {
            let history = file_history.lock();
            build_turn_diff(&history, turn_index, &root)
        };
        self.diff_viewer.set_turn_diff(files);
    }

    // -------------------------------------------------------------------
    // Event handling
    // -------------------------------------------------------------------

    /// Persist `has_completed_onboarding = true` to the settings file.
    /// Best-effort: failures are silently ignored to not disrupt the session.
    fn persist_onboarding_complete() -> anyhow::Result<()> {
        let mut settings = mangocode_core::config::Settings::load_sync()?;
        settings.has_completed_onboarding = true;
        settings.save_sync()
    }

    /// Process a keyboard event. Returns `true` when the input should be
    /// submitted (Enter pressed with no blocking dialog).
    pub fn handle_key_event(&mut self, key: KeyEvent) -> bool {
        let now = std::time::Instant::now();
        self.expire_paste_burst_if_idle(now);

        if self.global_search.open {
            return self.handle_global_search_key(key);
        }

        // ---- Context menu handling (highest priority for menu navigation) ----
        if self.context_menu_state.is_some() {
            match key.code {
                KeyCode::Esc => {
                    self.dismiss_context_menu();
                    return false;
                }
                KeyCode::Up | KeyCode::Down => {
                    self.navigate_context_menu(key.code);
                    return false;
                }
                KeyCode::Enter => {
                    self.execute_context_menu_item();
                    return false;
                }
                _ => {}
            }
        }

        // Bypass-permissions dialog: highest-priority gate — user must accept or the
        // session exits immediately. Mirrors TS BypassPermissionsModeDialog.tsx.
        if self.bypass_permissions_dialog.visible {
            match key.code {
                KeyCode::Char('1') | KeyCode::Esc => {
                    // "No, exit" — quit immediately
                    self.should_quit = true;
                }
                KeyCode::Char('2') => {
                    // "Yes, I accept" — dismiss and continue
                    self.bypass_permissions_dialog.dismiss();
                }
                KeyCode::Up | KeyCode::Char('k') => self.bypass_permissions_dialog.select_prev(),
                KeyCode::Down | KeyCode::Char('j') => self.bypass_permissions_dialog.select_next(),
                KeyCode::Enter => {
                    if self.bypass_permissions_dialog.is_accept_selected() {
                        self.bypass_permissions_dialog.dismiss();
                    } else {
                        self.should_quit = true;
                    }
                }
                _ => {}
            }
            return false;
        }

        // Onboarding dialog: shown on first launch, dismissed with Enter/→/Esc.
        if self.onboarding_dialog.visible {
            match key.code {
                KeyCode::Esc => {
                    self.onboarding_dialog.dismiss();
                }
                KeyCode::Enter | KeyCode::Right => {
                    if self.onboarding_dialog.next_page() {
                        self.onboarding_dialog.dismiss();
                        // Persist that onboarding is complete (best-effort).
                        let _ = Self::persist_onboarding_complete();
                    }
                }
                KeyCode::Left => {
                    self.onboarding_dialog.prev_page();
                }
                _ => {}
            }
            return false;
        }

        // Device code / browser auth dialog (GitHub Copilot, Anthropic OAuth)
        if self.device_auth_dialog.visible {
            match key.code {
                KeyCode::Esc => {
                    self.device_auth_dialog.close();
                }
                _ if matches!(
                    self.device_auth_dialog.status,
                    crate::device_auth_dialog::DeviceAuthStatus::Success(_)
                ) =>
                {
                    // Any key after success -> store credential and close
                    if let crate::device_auth_dialog::DeviceAuthStatus::Success(ref token) =
                        self.device_auth_dialog.status
                    {
                        let provider_id = self.device_auth_dialog.provider_id.clone();
                        let provider_name = self.device_auth_dialog.provider_name.clone();
                        let token = token.clone();
                        let credential = if provider_id == "github-copilot"
                            || provider_id == mangocode_core::ProviderId::ANTHROPIC_MAX
                        {
                            // OAuth flows store a Bearer token, not a plain API key.
                            mangocode_core::StoredCredential::OAuthToken {
                                access: token.clone(),
                                refresh: token,
                                expires: 0,
                            }
                        } else {
                            mangocode_core::StoredCredential::ApiKey { key: token }
                        };
                        self.auth_store.set(&provider_id, credential);
                        self.set_provider_default(provider_id.clone());
                        self.persist_provider_and_model();
                        self.has_credentials = true;
                        self.status_message = Some(format!(
                            "Connected to {}! Use /model to pick a model.",
                            provider_name
                        ));
                    }
                    self.device_auth_dialog.close();
                }
                _ if matches!(
                    self.device_auth_dialog.status,
                    crate::device_auth_dialog::DeviceAuthStatus::Error(_)
                ) =>
                {
                    // Any key after error -> close
                    self.device_auth_dialog.close();
                }
                _ => {} // Ignore other keys while waiting
            }
            return false;
        }

        // API key input dialog (opened from /connect for key-based providers)
        if self.key_input_dialog.visible {
            match key.code {
                KeyCode::Esc => {
                    self.key_input_dialog.close();
                }
                KeyCode::Enter => {
                    let provider_id = self.key_input_dialog.provider_id.clone();
                    let provider_name = self.key_input_dialog.provider_name.clone();
                    let pending_key = self.key_input_dialog.input.clone();
                    match validate_provider_credential(&provider_id, &pending_key) {
                        Ok(()) => {
                            let api_key = self.key_input_dialog.take_key();
                            self.auth_store.set(
                                &provider_id,
                                mangocode_core::StoredCredential::ApiKey { key: api_key },
                            );
                            self.set_provider_default(provider_id.clone());
                            self.persist_provider_and_model();
                            self.has_credentials = true;
                            self.status_message = Some(format!(
                                "Step 3/3 complete: connected to {}. Run /providers to verify live health, then /model to choose a model.",
                                provider_name
                            ));
                        }
                        Err((category, detail)) => {
                            self.status_message =
                                Some(format!("Step 2/3 failed [{}]: {}", category, detail));
                        }
                    }
                }
                KeyCode::Backspace => {
                    self.key_input_dialog.backspace();
                }
                KeyCode::Char(c) => {
                    self.key_input_dialog.insert_char(c);
                }
                _ => {}
            }
            return false;
        }

        // Connect-a-provider dialog (/connect command)
        if self.connect_dialog.visible {
            match key.code {
                KeyCode::Esc => {
                    self.connect_dialog.close();
                }
                KeyCode::Up => {
                    self.connect_dialog.move_up();
                }
                KeyCode::Down => {
                    self.connect_dialog.move_down();
                }
                KeyCode::PageUp => {
                    self.connect_dialog.page_up();
                }
                KeyCode::PageDown => {
                    self.connect_dialog.page_down();
                }
                KeyCode::Enter => {
                    if let Some(selected) = self.connect_dialog.selected().cloned() {
                        self.connect_dialog.close();
                        self.status_message = Some(format!(
                            "Step 1/3 complete: selected {}. Configure authentication next.",
                            selected.title
                        ));

                        match selected.id.as_str() {
                            // Local providers — activate immediately, no key needed
                            "ollama" | "lmstudio" | "llamacpp" => {
                                self.set_provider_default(selected.id.clone());
                                self.persist_provider_and_model();
                                self.has_credentials = true;
                                self.status_message = Some(format!(
                                    "Step 3/3 complete: switched to {} (local). Use /model to pick a model.",
                                    selected.title
                                ));
                            }
                            "anthropic" => {
                                // Anthropic: use API key from console.anthropic.com
                                self.key_input_dialog
                                    .open("anthropic".into(), "Anthropic (API Key)".into());
                                self.status_message = Some(format!(
                                    "Step 2/3: enter {} from {}.",
                                    get_env_var_for_provider("anthropic"),
                                    get_url_for_provider("anthropic")
                                ));
                            }
                            "anthropic-max" => {
                                // Claude Max: browser-based OAuth login with Claude.ai
                                self.device_auth_dialog
                                    .open("anthropic-max".into(), "Claude Max (OAuth)".into());
                                self.device_auth_pending = Some("anthropic-max".to_string());
                                self.status_message = Some(
                                    "Step 2/3: complete browser OAuth login with your Claude subscription. Step 3/3 will finish automatically."
                                        .to_string(),
                                );
                            }
                            "github-copilot" => {
                                // GitHub Copilot: device code flow with MangoCode's registered OAuth app
                                self.device_auth_dialog
                                    .open("github-copilot".into(), "GitHub Copilot".into());
                                self.device_auth_pending = Some("github-copilot".to_string());
                                self.status_message = Some(
                                    "Step 2/3: complete browser device login. Step 3/3 will finish automatically after token exchange."
                                        .to_string(),
                                );
                            }
                            // AWS Bedrock — accept a bearer token via key input dialog
                            "amazon-bedrock" => {
                                self.key_input_dialog.open(
                                    "amazon-bedrock".into(),
                                    "AWS Bedrock (Bearer Token)".into(),
                                );
                                self.status_message = Some(format!(
                                    "Step 2/3: enter {} (or your Bedrock bearer token).",
                                    get_env_var_for_provider("amazon-bedrock")
                                ));
                            }
                            // Google Vertex AI — auth comes from gcloud ADC.
                            // Show setup instructions and let the user configure env vars.
                            "google-vertex" => {
                                self.status_message = Some(
                                    "Step 2/3 (Vertex AI auth): set VERTEX_ENABLED=true, VERTEX_PROJECT_ID=<your-gcp-project>, optional VERTEX_LOCATION=us-central1, then run `gcloud auth application-default login` and restart MangoCode.\n\nStep 3/3: run /providers to verify connectivity. Or choose Google Vertex AI (Access Token) in /connect to enter VERTEX_ACCESS_TOKEN directly."
                                        .to_string(),
                                );
                            }
                            "google-vertex-access-token" => {
                                self.key_input_dialog.open(
                                    "google-vertex".into(),
                                    "Google Vertex AI (Access Token)".into(),
                                );
                                self.status_message = Some(
                                    "Step 2/3: enter VERTEX_ACCESS_TOKEN. Step 3/3: run /providers to verify connectivity."
                                        .to_string(),
                                );
                            }
                            // All other providers — open API key input dialog
                            _ => {
                                self.key_input_dialog
                                    .open(selected.id.clone(), selected.title.clone());
                                self.status_message = Some(format!(
                                    "Step 2/3: enter {} for {} ({}).",
                                    get_env_var_for_provider(&selected.id),
                                    selected.title,
                                    get_url_for_provider(&selected.id)
                                ));
                            }
                        }
                    }
                }
                KeyCode::Backspace => {
                    self.connect_dialog.filter_pop();
                }
                KeyCode::Char(c) => {
                    self.connect_dialog.filter_push(c);
                }
                _ => {}
            }
            return false;
        }

        // Command palette (Ctrl+K)
        if self.command_palette.visible {
            match key.code {
                KeyCode::Esc => {
                    self.command_palette.close();
                }
                KeyCode::Up => {
                    self.command_palette.move_up();
                }
                KeyCode::Down => {
                    self.command_palette.move_down();
                }
                KeyCode::PageUp => {
                    self.command_palette.page_up();
                }
                KeyCode::PageDown => {
                    self.command_palette.page_down();
                }
                KeyCode::Enter => {
                    if let Some(selected) = self.command_palette.selected().cloned() {
                        self.command_palette.close();
                        // Put the command in the input and signal for execution
                        self.prompt_input.replace_text(selected.id.clone());
                        return true; // signal to submit this as input
                    }
                }
                KeyCode::Backspace => {
                    self.command_palette.filter_pop();
                }
                KeyCode::Char(c) => {
                    self.command_palette.filter_push(c);
                }
                _ => {}
            }
            return false;
        }

        // Invalid-config dialog intercepts Enter/Esc to dismiss
        if self.invalid_config_dialog.visible {
            match key.code {
                KeyCode::Enter | KeyCode::Esc => self.invalid_config_dialog.dismiss(),
                KeyCode::Up => self.invalid_config_dialog.scroll_up(),
                KeyCode::Down => self.invalid_config_dialog.scroll_down(20),
                _ => {}
            }
            return false;
        }

        // Model picker intercepts navigation and Esc
        if self.model_picker.visible {
            match key.code {
                KeyCode::Esc => self.model_picker.close(),
                KeyCode::Up => self.model_picker.select_prev(),
                KeyCode::Down => self.model_picker.select_next(),
                KeyCode::Left => self.model_picker.effort_prev(),
                KeyCode::Right => self.model_picker.effort_next(),
                KeyCode::Enter => {
                    if let Some((model_id, effort)) = self.model_picker.confirm() {
                        // If user picked a model other than the fast-mode model
                        // while fast mode was active, turn fast mode off.
                        if self.fast_mode && model_id != FAST_MODE_MODEL {
                            self.fast_mode = false;
                        }
                        if let Some(e) = effort {
                            self.effort_level = e;
                        }
                        // Store explicit selections in the canonical
                        // "provider/model" form for non-Anthropic providers.
                        let provider = self.config.provider.as_deref().unwrap_or("anthropic");
                        let full_model = if provider == "anthropic" {
                            model_id.clone()
                        } else {
                            format!("{}/{}", provider, model_id)
                        };
                        self.set_model(full_model.clone());
                        self.persist_provider_and_model();
                        let effort_hint = effort
                            .map(|e| format!(" [{}]", e.label()))
                            .unwrap_or_default();
                        self.status_message = Some(format!("Model: {}{}", full_model, effort_hint));
                    }
                }
                KeyCode::Backspace => self.model_picker.pop_filter_char(),
                KeyCode::Char(c) => self.model_picker.push_filter_char(c),
                _ => {}
            }
            return false;
        }

        // Session branching overlay intercepts navigation and Esc
        if self.session_branching.visible {
            use crate::session_branching::BranchBrowserMode;
            match self.session_branching.mode {
                BranchBrowserMode::Browse => match key.code {
                    KeyCode::Esc => self.session_branching.cancel(),
                    KeyCode::Up => self.session_branching.select_prev(),
                    KeyCode::Down => self.session_branching.select_next(),
                    KeyCode::Char('n') => self.session_branching.start_create_new(),
                    KeyCode::Char('d') => self.session_branching.start_delete_confirm(),
                    KeyCode::Enter => {
                        if let Some(branch) = self.session_branching.selected_branch() {
                            self.status_message =
                                Some(format!("Switched to branch: {}", branch.name));
                            self.session_branching.close();
                        }
                    }
                    _ => {}
                },
                BranchBrowserMode::CreateNew => match key.code {
                    KeyCode::Esc => self.session_branching.cancel(),
                    KeyCode::Enter => {
                        if let Some((name, at_msg)) = self.session_branching.confirm_create_new() {
                            self.status_message =
                                Some(format!("Created branch: {} at message {}", name, at_msg));
                            self.session_branching.close();
                        }
                    }
                    KeyCode::Backspace => self.session_branching.pop_create_char(),
                    KeyCode::Char(c) => self.session_branching.push_create_char(c),
                    _ => {}
                },
                BranchBrowserMode::ConfirmDelete => match key.code {
                    KeyCode::Esc | KeyCode::Char('n') => self.session_branching.cancel(),
                    KeyCode::Enter | KeyCode::Char('y') => {
                        if let Some(branch_id) = self.session_branching.confirm_delete() {
                            self.status_message = Some(format!("Deleted branch: {}", branch_id));
                        }
                    }
                    _ => {}
                },
            }
            return false;
        }

        // Session browser intercepts navigation and Esc
        if self.session_browser.visible {
            use crate::session_browser::SessionBrowserMode;
            match self.session_browser.mode {
                SessionBrowserMode::Browse => match key.code {
                    KeyCode::Esc => self.session_browser.close(),
                    KeyCode::Up => self.session_browser.select_prev(),
                    KeyCode::Down => self.session_browser.select_next(),
                    KeyCode::Char('r') => self.session_browser.start_rename(),
                    _ => {}
                },
                SessionBrowserMode::Rename => match key.code {
                    KeyCode::Esc => self.session_browser.cancel(),
                    KeyCode::Enter => {
                        if let Some((_id, name)) = self.session_browser.confirm_rename() {
                            self.session_title = Some(name.clone());
                            self.status_message = Some(format!("Renamed to: {}", name));
                        }
                    }
                    KeyCode::Backspace => self.session_browser.pop_rename_char(),
                    KeyCode::Char(c) => self.session_browser.push_rename_char(c),
                    _ => {}
                },
                SessionBrowserMode::Confirm => match key.code {
                    KeyCode::Esc | KeyCode::Char('n') => self.session_browser.cancel(),
                    KeyCode::Enter | KeyCode::Char('y') => {
                        self.session_browser.close();
                    }
                    _ => {}
                },
            }
            return false;
        }

        // Tasks overlay intercepts navigation and Esc
        if self.tasks_overlay.visible {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.tasks_overlay.close(),
                KeyCode::Up => self.tasks_overlay.select_prev(),
                KeyCode::Down => self.tasks_overlay.select_next(),
                KeyCode::Enter => {
                    if let Some((task_id, new_status)) =
                        self.tasks_overlay.cycle_and_persist_status()
                    {
                        self.status_message = Some(format!("Task {} → {}", task_id, new_status));
                    }
                }
                _ => {}
            }
            return false;
        }

        // Export dialog key handling
        if self.export_dialog.visible {
            match key.code {
                KeyCode::Esc => {
                    self.export_dialog.dismiss();
                }
                KeyCode::Enter => {
                    if let Some(path) = self.perform_export() {
                        self.notifications.push(
                            NotificationKind::Info,
                            format!("Exported to {}", path),
                            Some(4),
                        );
                    } else {
                        self.notifications.push(
                            NotificationKind::Warning,
                            "Export failed: could not write file.".to_string(),
                            Some(4),
                        );
                    }
                }
                KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
                    self.export_dialog.toggle();
                }
                KeyCode::Char('1') => {
                    self.export_dialog.selected = ExportFormat::Json;
                }
                KeyCode::Char('2') => {
                    self.export_dialog.selected = ExportFormat::Markdown;
                }
                _ => {}
            }
            return false;
        }

        // Context visualization overlay key handling
        if self.context_viz.visible {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    self.context_viz.close();
                }
                _ => {}
            }
            return false;
        }

        // MCP approval dialog
        if self.mcp_approval.visible {
            let result = crate::dialogs::handle_mcp_approval_key(&mut self.mcp_approval, key);
            if result.is_some() {
                // Result processed by CLI loop via take_mcp_approval_result()
            }
            return false;
        }

        // Feedback survey intercepts digit keys and Esc
        if self.feedback_survey.visible {
            if key.code == KeyCode::Esc {
                self.feedback_survey.close();
                return false;
            }
            if let KeyCode::Char(c) = key.code {
                if let Some(d) = c.to_digit(10) {
                    self.feedback_survey.handle_digit(d as u8);
                    return false;
                }
            }
            return false;
        }

        // Memory file selector intercepts navigation and Esc
        if self.memory_file_selector.visible {
            match key.code {
                KeyCode::Esc => self.memory_file_selector.close(),
                KeyCode::Up => self.memory_file_selector.select_prev(),
                KeyCode::Down => self.memory_file_selector.select_next(),
                KeyCode::Enter => {
                    // Selection acknowledged — consumer can read selected_path()
                    self.memory_file_selector.close();
                }
                _ => {}
            }
            return false;
        }

        // Hooks config menu intercepts navigation and Esc
        if self.hooks_config_menu.visible {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.hooks_config_menu.back(),
                KeyCode::Enter => self.hooks_config_menu.enter(),
                KeyCode::Up | KeyCode::Char('k') => self.hooks_config_menu.select_prev(),
                KeyCode::Down | KeyCode::Char('j') => self.hooks_config_menu.select_next(),
                _ => {}
            }
            return false;
        }

        if self.diff_viewer.open {
            self.handle_diff_viewer_key(key);
            return false;
        }

        if self.agents_menu.open {
            self.handle_agents_menu_key(key);
            return false;
        }

        if self.mcp_view.open {
            self.handle_mcp_view_key(key);
            return false;
        }

        if self.stats_dialog.open {
            self.handle_stats_dialog_key(key);
            return false;
        }

        // Settings screen intercepts keys
        if self.settings_screen.visible {
            crate::settings_screen::handle_settings_key(
                &mut self.settings_screen,
                &mut self.config,
                key,
            );
            return false;
        }

        // Theme picker intercepts keys
        if self.theme_screen.visible {
            if let Some(theme_name) =
                crate::theme_screen::handle_theme_key(&mut self.theme_screen, key)
            {
                self.apply_theme(&theme_name);
            }
            return false;
        }

        // Privacy screen intercepts keys
        if self.privacy_screen.visible {
            crate::privacy_screen::handle_privacy_key(&mut self.privacy_screen, key);
            return false;
        }

        // Rewind flow overlay intercepts keys first
        if self.rewind_flow.visible {
            return self.handle_rewind_flow_key(key);
        }

        // Help overlay intercepts keys next
        if self.help_overlay.visible {
            return self.handle_help_overlay_key(key);
        }

        // New history-search overlay
        if self.history_search_overlay.visible {
            return self.handle_history_search_overlay_key(key);
        }

        if self.global_search.open {
            return self.handle_global_search_key(key);
        }

        // Legacy history-search mode intercepts most keys
        if self.history_search.is_some() {
            return self.handle_history_search_key(key);
        }

        // Permission dialog mode intercepts most keys
        if self.permission_request.is_some() {
            self.handle_permission_key(key);
            return false;
        }

        // ESC should always stop an active stream.
        if key.code == KeyCode::Esc && self.is_streaming {
            self.is_streaming = false;
            self.spinner_verb = None;
            self.streaming_text.clear();
            self.tool_use_blocks.clear();
            self.status_message = Some("Cancelled.".to_string());
            return false;
        }

        // Notification dismiss
        if key.code == KeyCode::Esc && !self.notifications.is_empty() {
            self.notifications.dismiss_current();
            return false;
        }

        // Plugin hint dismiss
        if key.code == KeyCode::Esc {
            if let Some(hint) = self.plugin_hints.iter_mut().find(|h| h.is_visible()) {
                hint.dismiss();
                return false;
            }
        }

        // Overage upsell dismiss
        if key.code == KeyCode::Esc && self.overage_upsell.visible {
            self.overage_upsell.dismiss();
            return false;
        }

        // Voice mode notice dismiss
        if key.code == KeyCode::Esc && self.voice_mode_notice.visible {
            self.voice_mode_notice.dismiss();
            return false;
        }

        // Desktop upsell startup dialog
        if self.desktop_upsell.visible {
            match key.code {
                KeyCode::Up | KeyCode::BackTab => {
                    self.desktop_upsell.select_prev();
                    return false;
                }
                KeyCode::Down | KeyCode::Tab => {
                    self.desktop_upsell.select_next();
                    return false;
                }
                KeyCode::Enter => {
                    self.desktop_upsell.confirm();
                    return false;
                }
                KeyCode::Esc => {
                    self.desktop_upsell.dismiss_temporarily();
                    return false;
                }
                _ => return false,
            }
        }

        // Memory update notification dismiss
        if key.code == KeyCode::Esc && self.memory_update_notification.visible {
            self.memory_update_notification.dismiss();
            return false;
        }

        // MCP elicitation dialog — highest priority modal
        if self.elicitation.visible {
            match key.code {
                KeyCode::Esc => {
                    self.elicitation.cancel();
                    return false;
                }
                KeyCode::Enter => {
                    self.elicitation.submit();
                    return false;
                }
                KeyCode::Tab | KeyCode::Down => {
                    if let crossterm::event::KeyModifiers::SHIFT = key.modifiers {
                        self.elicitation.prev_field();
                    } else {
                        self.elicitation.next_field();
                    }
                    return false;
                }
                KeyCode::BackTab | KeyCode::Up => {
                    self.elicitation.prev_field();
                    return false;
                }
                KeyCode::Left => {
                    self.elicitation.cycle_enum_prev();
                    return false;
                }
                KeyCode::Right => {
                    self.elicitation.cycle_enum_next();
                    return false;
                }
                KeyCode::Char(' ') => {
                    self.elicitation.toggle_active();
                    return false;
                }
                KeyCode::Backspace => {
                    self.elicitation.backspace();
                    return false;
                }
                KeyCode::Char(c) => {
                    self.elicitation.insert_char(c);
                    return false;
                }
                _ => return false,
            }
        }

        // Shift+Enter and active paste-burst must never route through submit keybindings.
        if key.code == KeyCode::Enter
            && !self.is_streaming
            && (key.modifiers.contains(KeyModifiers::SHIFT) || self.is_in_paste_burst(now))
        {
            self.prompt_input.insert_newline();
            self.refresh_prompt_input();
            self.activate_paste_burst(now);
            return false;
        }

        // ---- Keybinding processor (runs AFTER all dialog checks) ----------
        let key_context = self.current_key_context();
        if let Some(keystroke) = key_event_to_keystroke(&key) {
            let had_pending_chord = self.keybindings.has_pending_chord();
            match self.keybindings.process(keystroke, &key_context) {
                KeybindingResult::Action(action) => {
                    return self.handle_keybinding_action(&action);
                }
                KeybindingResult::Unbound | KeybindingResult::Pending => return false,
                KeybindingResult::NoMatch if had_pending_chord => return false,
                KeybindingResult::NoMatch => {}
            }
        } else {
            self.keybindings.cancel_chord();
        }

        // Clear any active text selection on key press (except Ctrl+C which copies it).
        let is_copy =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if !is_copy && self.selection_anchor.is_some() {
            self.selection_anchor = None;
            self.selection_focus = None;
            *self.selection_text.borrow_mut() = String::new();
        }

        // ---- Voice hold-to-talk (Alt+V toggles recording on/off) ----------
        if key.code == KeyCode::Char('v')
            && key.modifiers.contains(KeyModifiers::ALT)
            && self.voice_recorder.is_some()
        {
            if !self.voice_recording {
                // First press: start recording.
                let (tx, rx) = tokio::sync::mpsc::channel(8);
                self.voice_event_rx = Some(rx);
                self.voice_recording = true;
                if let Some(ref recorder_arc) = self.voice_recorder {
                    let recorder = recorder_arc.clone();
                    // Use spawn_blocking so we don't hold a std::sync::MutexGuard
                    // across an await point.  start_recording internally spawns a
                    // tokio task and returns quickly, so blocking is negligible.
                    tokio::task::spawn_blocking(move || {
                        if let Ok(mut r) = recorder.lock() {
                            // start_recording is async but its real work happens in
                            // a spawned task; use block_on to drive the short setup.
                            tokio::runtime::Handle::current()
                                .block_on(r.start_recording(tx))
                                .ok();
                        }
                    });
                }
                self.notifications.push(
                    NotificationKind::Info,
                    "Recording\u{2026} (press Alt+V again to transcribe)".to_string(),
                    None,
                );
            } else {
                // Second press: stop recording.  stop_recording() just flips an
                // AtomicBool; drive it synchronously to avoid Send issues.
                self.voice_recording = false;
                if let Some(ref recorder_arc) = self.voice_recorder {
                    let recorder = recorder_arc.clone();
                    tokio::task::spawn_blocking(move || {
                        if let Ok(mut r) = recorder.lock() {
                            tokio::runtime::Handle::current()
                                .block_on(r.stop_recording())
                                .ok();
                        }
                    });
                }
                self.notifications.push(
                    NotificationKind::Info,
                    "Transcribing\u{2026}".to_string(),
                    Some(10),
                );
            }
            return false;
        }

        // ---- Voice PTT: plain V press starts recording when voice is on ----
        // This is the "hold to talk" variant.  The user presses V to begin
        // recording; releasing V (handled in the run loop) or pressing Enter
        // stops the capture and triggers transcription.
        // Only active when voice mode is enabled (voice_recorder is Some) and
        // the prompt input is in default (non-vim) mode so 'v' doesn't conflict
        // with vim keybindings.
        if key.code == KeyCode::Char('v')
            && key.modifiers == KeyModifiers::NONE
            && self.voice_recorder.is_some()
            && !self.voice_recording
            && self.prompt_input.vim_mode == crate::prompt_input::VimMode::Insert
        {
            self.handle_voice_ptt_start();
            return false;
        }

        // ---- Ctrl+V — clipboard paste (image first, then text fallback) ----
        // Only fires when NOT in vim Normal/Visual/VisualBlock mode (where \x16 is
        // already consumed by the vim handler above to enter VisualBlock mode).
        if key.code == KeyCode::Char('v')
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && !matches!(
                self.prompt_input.vim_mode,
                crate::prompt_input::VimMode::Normal
                    | crate::prompt_input::VimMode::Visual
                    | crate::prompt_input::VimMode::VisualBlock
            )
        {
            use crate::image_paste::{read_clipboard_image, read_clipboard_text};
            if let Some(img) = read_clipboard_image() {
                let label = img.label.clone();
                let dims = img.dimensions;
                self.prompt_input.add_image(img);
                let msg = if let Some((w, h)) = dims {
                    format!("Image attached: {} ({}x{})", label, w, h)
                } else {
                    format!("Image attached: {}", label)
                };
                self.notifications
                    .push(NotificationKind::Info, msg, Some(3));
            } else if let Some(text) = read_clipboard_text() {
                self.handle_prompt_paste(&text, now);
            }
            return false;
        }

        // ---- Enter while PTT recording: stop capture instead of submitting ----
        if key.code == KeyCode::Enter && self.voice_recording && self.voice_recorder.is_some() {
            self.handle_voice_ptt_stop();
            return false;
        }

        // ---- Focus state machine: transcript mode --------------------------
        // When the transcript pane has focus, intercept Escape and scroll keys.
        // Printable characters switch focus back to Input and fall through so the
        // keystroke is processed normally by the prompt editor below.
        if self.focus == FocusTarget::Transcript {
            match key.code {
                KeyCode::Esc => {
                    self.focus = FocusTarget::Input;
                    return false;
                }
                KeyCode::PageUp | KeyCode::PageDown => {
                    // Let these fall through to the normal scroll handling below.
                }
                KeyCode::Char(_)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    // Printable char: switch focus to Input and process normally.
                    self.focus = FocusTarget::Input;
                }
                _ => {}
            }
        }

        match key.code {
            // ---- Quit / cancel ----------------------------------------
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // If text is selected, copy it to clipboard instead of quitting.
                let sel_text = self.selection_text.borrow().clone();
                if self.selection_anchor.is_some() && !sel_text.is_empty() {
                    let copied = crate::image_paste::write_clipboard_text(&sel_text);
                    self.selection_anchor = None;
                    self.selection_focus = None;
                    *self.selection_text.borrow_mut() = String::new();
                    if copied {
                        self.notifications.push(
                            NotificationKind::Info,
                            "Copied to clipboard".to_string(),
                            Some(2),
                        );
                    }
                } else if self.is_streaming {
                    self.is_streaming = false;
                    self.spinner_verb = None;
                    self.streaming_text.clear();
                    self.tool_use_blocks.clear();
                    self.status_message = Some("Cancelled.".to_string());
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.prompt_input.is_empty() {
                    self.should_quit = true;
                }
            }

            // ---- History search ----------------------------------------
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Open the new overlay-based history search
                let overlay = HistorySearchOverlay::open(&self.prompt_input.history);
                self.history_search_overlay = overlay;
                // Also open legacy for backwards compat
                let mut hs = HistorySearch::new();
                hs.update_matches(&self.prompt_input.history);
                self.history_search = Some(hs);
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.global_search.open();
                self.refresh_global_search();
            }

            // ---- Tasks overlay (Ctrl+T) --------------------------------
            KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.tasks_overlay.toggle();
            }

            // ---- Help overlay ------------------------------------------
            KeyCode::F(1) => {
                self.show_help = !self.show_help;
                self.help_overlay.toggle();
            }

            // ---- Ctrl+A: Open model picker ----
            KeyCode::Char('a')
                if key.modifiers.contains(KeyModifiers::CONTROL) && !self.is_streaming =>
            {
                self.intercept_slash_command("model");
            }

            // ---- Ctrl+K: Command palette (or Emacs kill-line if input has text) ----
            KeyCode::Char('k')
                if key.modifiers.contains(KeyModifiers::CONTROL) && !self.is_streaming =>
            {
                if self.prompt_input.is_empty() {
                    self.command_palette.open();
                } else {
                    self.prompt_input.kill_line();
                    self.refresh_prompt_input();
                }
            }
            KeyCode::Char('u')
                if key.modifiers.contains(KeyModifiers::CONTROL) && !self.is_streaming =>
            {
                self.prompt_input.kill_line_backward();
                self.refresh_prompt_input();
            }
            KeyCode::Char('w')
                if key.modifiers.contains(KeyModifiers::CONTROL) && !self.is_streaming =>
            {
                self.prompt_input.kill_word_backward();
                self.refresh_prompt_input();
            }
            KeyCode::Char('y')
                if key.modifiers.contains(KeyModifiers::CONTROL) && !self.is_streaming =>
            {
                self.prompt_input.yank();
                self.refresh_prompt_input();
            }

            // ---- Alt/Meta key text editing operations -------------------
            KeyCode::Char('y')
                if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming =>
            {
                self.prompt_input.yank_pop();
                self.refresh_prompt_input();
            }
            KeyCode::Backspace
                if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming =>
            {
                self.prompt_input.delete_word_backward();
                self.refresh_prompt_input();
            }
            KeyCode::Delete if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming => {
                self.prompt_input.delete_word_forward();
                self.refresh_prompt_input();
            }
            KeyCode::Char('b')
                if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming =>
            {
                self.prompt_input.move_word_backward();
                self.sync_legacy_prompt_fields();
            }
            KeyCode::Char('f')
                if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming =>
            {
                self.prompt_input.move_word_forward();
                self.sync_legacy_prompt_fields();
            }
            KeyCode::Char('d')
                if key.modifiers.contains(KeyModifiers::ALT) && !self.is_streaming =>
            {
                self.prompt_input.delete_word_at_cursor();
                self.refresh_prompt_input();
            }

            // ---- Text entry (blocked while streaming) ------------------
            KeyCode::Char(c) if !self.is_streaming => {
                if self.prompt_input.vim_enabled && self.prompt_input.vim_mode != VimMode::Insert {
                    self.prompt_input.vim_command(&c.to_string());
                } else {
                    self.prompt_input.insert_char(c);
                    self.register_printable_keystroke_for_paste_burst(key, now);
                }
                self.refresh_prompt_input();
            }
            KeyCode::Backspace if !self.is_streaming => {
                self.prompt_input.backspace();
                self.refresh_prompt_input();
            }
            KeyCode::Delete if !self.is_streaming => {
                self.prompt_input.delete();
                self.refresh_prompt_input();
            }
            KeyCode::Left if !self.is_streaming => {
                self.prompt_input.move_left();
                self.sync_legacy_prompt_fields();
            }
            KeyCode::Right if !self.is_streaming => {
                self.prompt_input.move_right();
                self.sync_legacy_prompt_fields();
            }
            KeyCode::Home if !self.is_streaming => {
                self.prompt_input.cursor = 0;
                self.sync_legacy_prompt_fields();
            }
            KeyCode::End if !self.is_streaming => {
                self.prompt_input.cursor = self.prompt_input.text.len();
                self.sync_legacy_prompt_fields();
            }
            KeyCode::Tab if !self.is_streaming => {
                if !self.prompt_input.suggestions.is_empty() {
                    if self.prompt_input.suggestion_index.is_none() {
                        self.prompt_input.suggestion_index = Some(0);
                    }
                    self.prompt_input.accept_suggestion();
                    self.refresh_prompt_input();
                }
            }

            // ---- Shift+Tab: cycle permission mode ----------------------
            // Default → AcceptEdits → BypassPermissions → Default
            // Mirrors TS bottom-left indicator cycling behaviour.
            KeyCode::BackTab if !self.is_streaming => {
                use mangocode_core::config::PermissionMode;
                self.config.permission_mode = match self.config.permission_mode {
                    PermissionMode::Default => PermissionMode::AcceptEdits,
                    PermissionMode::AcceptEdits => PermissionMode::BypassPermissions,
                    PermissionMode::BypassPermissions => PermissionMode::Default,
                    PermissionMode::Plan => PermissionMode::Default,
                };
                let label = match self.config.permission_mode {
                    PermissionMode::Default => "Default permissions",
                    PermissionMode::AcceptEdits => "Accept-edits mode",
                    PermissionMode::BypassPermissions => "Bypass permissions (dangerous)",
                    PermissionMode::Plan => "Plan mode",
                };
                self.status_message = Some(label.to_string());
            }

            // ---- Submit ------------------------------------------------
            KeyCode::Enter if !self.is_streaming => {
                if key.modifiers.contains(KeyModifiers::SHIFT) || self.is_in_paste_burst(now) {
                    self.prompt_input.insert_newline();
                    self.refresh_prompt_input();
                    self.activate_paste_burst(now);
                    return false;
                }

                // If a slash-command suggestion is selected, accept it instead of submitting.
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.suggestion_index.is_some()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.accept_suggestion();
                    self.refresh_prompt_input();
                    return false;
                }
                // New user input: snap back to bottom.
                self.auto_scroll = true;
                self.new_messages_while_scrolled = 0;
                self.scroll_offset = 0;
                return true;
            }

            // ---- Message boundary navigation (Alt+Up/Alt+Down) ----------
            KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => {
                // Jump up by ~20 lines (approximate message boundary).
                self.scroll_offset = self.scroll_offset.saturating_add(20);
                self.auto_scroll = false;
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => {
                // Jump down by ~20 lines (approximate message boundary).
                let new_off = self.scroll_offset.saturating_sub(20);
                self.scroll_offset = new_off;
                if new_off == 0 {
                    self.auto_scroll = true;
                    self.new_messages_while_scrolled = 0;
                }
            }

            // ---- Input history navigation ------------------------------
            KeyCode::Up => {
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.suggestion_prev();
                } else if !self.prompt_input.history.is_empty() {
                    self.prompt_input.history_up();
                }
                self.refresh_prompt_input();
            }
            KeyCode::Down => {
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.suggestion_next();
                } else if self.prompt_input.history_pos.is_some() {
                    self.prompt_input.history_down();
                }
                self.refresh_prompt_input();
            }

            // ---- Scroll ------------------------------------------------
            KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
                // Scrolling up disables auto-follow.
                self.auto_scroll = false;
            }
            KeyCode::PageDown => {
                let new_off = self.scroll_offset.saturating_sub(10);
                self.scroll_offset = new_off;
                if new_off == 0 {
                    // Scrolled all the way back to bottom — re-enable auto-follow.
                    self.auto_scroll = true;
                    self.new_messages_while_scrolled = 0;
                }
            }

            // ---- Toggle last thinking block (t key) -------------------
            KeyCode::Char('t') if !self.is_streaming => {
                // Find the last thinking block in the message list and toggle it
                use mangocode_core::types::ContentBlock;
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                'outer: for msg in self.messages.iter().rev() {
                    let blocks = msg.content_blocks();
                    for block in blocks.iter().rev() {
                        if let ContentBlock::Thinking { thinking, .. } = block {
                            let mut h = DefaultHasher::new();
                            thinking.hash(&mut h);
                            let hash = h.finish();
                            if self.thinking_expanded.contains(&hash) {
                                self.thinking_expanded.remove(&hash);
                            } else {
                                self.thinking_expanded.insert(hash);
                            }
                            self.invalidate_transcript();
                            break 'outer;
                        }
                    }
                }
            }

            _ => {}
        }
        false
    }

    fn current_key_context(&self) -> KeyContext {
        if self.diff_viewer.open {
            KeyContext::DiffDialog
        } else if self.agents_menu.open || self.mcp_view.open || self.stats_dialog.open {
            KeyContext::Select
        } else if self.settings_screen.visible {
            KeyContext::Settings
        } else if self.theme_screen.visible {
            KeyContext::ThemePicker
        } else if self.rewind_flow.visible {
            KeyContext::Confirmation
        } else if self.help_overlay.visible {
            KeyContext::Help
        } else if self.history_search_overlay.visible || self.history_search.is_some() {
            KeyContext::HistorySearch
        } else if self.permission_request.is_some() {
            KeyContext::Confirmation
        } else if self.show_help {
            KeyContext::Help
        } else {
            KeyContext::Chat
        }
    }

    // -------------------------------------------------------------------
    // New overlay key handlers
    // -------------------------------------------------------------------

    fn handle_stats_dialog_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.stats_dialog.close(),
            KeyCode::Tab | KeyCode::Right => self.stats_dialog.next_tab(),
            KeyCode::BackTab | KeyCode::Left => self.stats_dialog.prev_tab(),
            KeyCode::Char('r') => self.stats_dialog.cycle_range(),
            KeyCode::Up => self.stats_dialog.scroll = self.stats_dialog.scroll.saturating_sub(1),
            KeyCode::Down => self.stats_dialog.scroll = self.stats_dialog.scroll.saturating_add(1),
            _ => {}
        }
    }

    fn handle_mcp_view_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.mcp_view.close(),
            KeyCode::Tab | KeyCode::Left | KeyCode::Right => self.mcp_view.switch_pane(),
            KeyCode::Up => self.mcp_view.select_prev(),
            KeyCode::Down => self.mcp_view.select_next(),
            KeyCode::Backspace => self.mcp_view.pop_search_char(),
            KeyCode::Char('e') => self.mcp_view.toggle_error_detail(),
            KeyCode::Char('r') => {
                self.pending_mcp_reconnect = true;
                self.status_message = Some("Reconnecting MCP runtime...".to_string());
            }
            KeyCode::Char(c) if key.modifiers.is_empty() => {
                if self.mcp_view.active_pane != crate::mcp_view::McpViewPane::ServerList {
                    self.mcp_view.push_search_char(c);
                }
            }
            _ => {}
        }
    }

    fn handle_agents_menu_key(&mut self, key: KeyEvent) {
        if matches!(self.agents_menu.route, AgentsRoute::Editor(_)) {
            match key.code {
                KeyCode::Esc => self.agents_menu.go_back(),
                KeyCode::Tab | KeyCode::Down => self.agents_menu.editor_next_field(),
                KeyCode::BackTab | KeyCode::Up => self.agents_menu.editor_prev_field(),
                KeyCode::Enter => self.agents_menu.editor_insert_newline(),
                KeyCode::Backspace => self.agents_menu.editor_backspace(),
                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    match self.agents_menu.save_editor() {
                        Ok(msg) => self.status_message = Some(msg),
                        Err(err) => {
                            self.agents_menu.editor.error = Some(err.clone());
                            self.agents_menu.editor.saved_message = None;
                            self.status_message = Some(err);
                        }
                    }
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.agents_menu.editor_insert_char(ch);
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Backspace => self.agents_menu.go_back(),
            KeyCode::Up => self.agents_menu.select_prev(),
            KeyCode::Down => self.agents_menu.select_next(),
            KeyCode::Enter | KeyCode::Right => self.agents_menu.confirm_selection(),
            KeyCode::Left => self.agents_menu.go_back(),
            _ => {}
        }
    }

    fn handle_diff_viewer_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.diff_viewer.close(),
            KeyCode::Tab | KeyCode::Left | KeyCode::Right => self.diff_viewer.switch_pane(),
            KeyCode::Char('d') => {
                let root = self.project_root();
                self.diff_viewer.toggle_diff_type(&root);
            }
            KeyCode::Up => {
                if self.diff_viewer.active_pane == DiffPane::FileList {
                    self.diff_viewer.select_prev();
                } else {
                    self.diff_viewer.scroll_detail_up();
                }
            }
            KeyCode::Down => {
                if self.diff_viewer.active_pane == DiffPane::FileList {
                    self.diff_viewer.select_next();
                } else {
                    self.diff_viewer.scroll_detail_down();
                }
            }
            KeyCode::PageUp => self.diff_viewer.scroll_detail_up(),
            KeyCode::PageDown => self.diff_viewer.scroll_detail_down(),
            KeyCode::Char(' ') => {
                if self.diff_viewer.active_pane == DiffPane::FileList {
                    self.diff_viewer.toggle_file_collapse();
                }
            }
            _ => {}
        }
    }

    fn handle_help_overlay_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc | KeyCode::F(1) => {
                self.help_overlay.close();
                self.show_help = false;
            }
            KeyCode::Char('?') if key.modifiers.is_empty() => {
                self.help_overlay.close();
                self.show_help = false;
            }
            KeyCode::Up => {
                self.help_overlay.scroll_up();
            }
            KeyCode::Down => {
                let max = 50u16; // generous upper bound; renderer will clamp
                self.help_overlay.scroll_down(max);
            }
            KeyCode::Backspace => {
                self.help_overlay.pop_filter_char();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.help_overlay.push_filter_char(c);
            }
            _ => {}
        }
        false
    }

    fn handle_history_search_overlay_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                self.history_search_overlay.close();
                self.history_search = None;
            }
            KeyCode::Enter => {
                if let Some(entry) = self
                    .history_search_overlay
                    .current_entry(&self.prompt_input.history)
                {
                    self.set_prompt_text(entry.to_string());
                }
                self.history_search_overlay.close();
                self.history_search = None;
            }
            KeyCode::Up => {
                self.history_search_overlay.select_prev();
                if let Some(hs) = self.history_search.as_mut() {
                    if hs.selected > 0 {
                        hs.selected -= 1;
                    }
                }
            }
            KeyCode::Down => {
                self.history_search_overlay.select_next();
                if let Some(hs) = self.history_search.as_mut() {
                    let max = hs.matches.len().saturating_sub(1);
                    if hs.selected < max {
                        hs.selected += 1;
                    }
                }
            }
            KeyCode::Backspace => {
                let history = self.prompt_input.history.clone();
                self.history_search_overlay.pop_char(&history);
                if let Some(hs) = self.history_search.as_mut() {
                    hs.query.pop();
                    hs.update_matches(&history);
                }
            }
            // 'p' with no modifiers and an empty query = pin/unpin the selected entry.
            // When the query is non-empty 'p' is treated as a filter character so
            // the user can still search for prompts containing the letter 'p'.
            KeyCode::Char('p')
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.history_search_overlay.query.is_empty() =>
            {
                self.history_search_overlay.toggle_pin();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                let history = self.prompt_input.history.clone();
                self.history_search_overlay.push_char(c, &history);
                if let Some(hs) = self.history_search.as_mut() {
                    hs.query.push(c);
                    hs.update_matches(&history);
                }
            }
            _ => {}
        }
        false
    }

    fn handle_rewind_flow_key(&mut self, key: KeyEvent) -> bool {
        use crate::overlays::RewindStep;
        match &self.rewind_flow.step {
            RewindStep::Selecting => match key.code {
                KeyCode::Esc => {
                    self.rewind_flow.close();
                }
                KeyCode::Enter => {
                    self.rewind_flow.confirm_selection();
                }
                KeyCode::Up => {
                    self.rewind_flow.selector.select_prev();
                }
                KeyCode::Down => {
                    self.rewind_flow.selector.select_next();
                }
                _ => {}
            },
            RewindStep::Confirming { .. } => match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    if let Some(idx) = self.rewind_flow.accept_confirm() {
                        // Truncate conversation to the selected message index.
                        self.messages.truncate(idx);
                        // Remove system annotations placed after the truncation point.
                        self.system_annotations.retain(|a| a.after_index <= idx);
                        self.notifications.push(
                            NotificationKind::Success,
                            format!("Rewound to message #{}", idx),
                            Some(4),
                        );
                    }
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.rewind_flow.reject_confirm();
                }
                _ => {}
            },
        }
        false
    }

    fn handle_global_search_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Esc => {
                self.global_search.close();
            }
            KeyCode::Enter => {
                if let Some(selected) = self.global_search.selected_ref() {
                    self.set_prompt_text(selected);
                }
                self.global_search.close();
            }
            KeyCode::Up => self.global_search.select_prev(),
            KeyCode::Down => self.global_search.select_next(),
            KeyCode::Backspace => {
                self.global_search.pop_char();
                self.refresh_global_search();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.global_search.push_char(c);
                self.refresh_global_search();
            }
            _ => {}
        }
        false
    }

    fn canonical_key_action(action: &str) -> &str {
        match action {
            // New namespaced vocabulary
            "session.abort" => "interrupt",
            "session.close" => "exit",
            "view.repaint" => "redraw",
            "history.overlay.open" => "historySearch",
            "chat.submit" => "submit",
            "history.entry.prev" => "historyPrev",
            "history.entry.next" => "historyNext",
            "transcript.scroll.up" => "scrollUp",
            "transcript.scroll.down" => "scrollDown",
            "suggestion.accept" => "indent",
            "search.global.open" => "globalSearch",
            "search.transcript.open" => "findInMessage",
            "search.transcript.next" => "findNext",
            "search.transcript.prev" => "findPrev",
            "confirm.accept" => "yes",
            "confirm.reject" => "no",
            "confirm.prev" => "prevOption",
            "confirm.next" => "nextOption",
            "overlay.dismiss" => "close",
            "history.result.select" => "select",
            "history.result.cancel" => "cancel",
            "history.result.prev" => "prevResult",
            "history.result.next" => "nextResult",
            "input.clear" => "clearLine",
            "transcript.issue.next" => "jumpToNextError",
            "transcript.issue.prev" => "jumpToPreviousError",
            "transcript.jump.prev" => "previousMessage",
            "transcript.jump.next" => "nextMessage",
            "mode.cycle" => "reverseIndent",
            "chat.submit.alt" => "sendMessage",
            _ => action,
        }
    }

    fn handle_keybinding_action(&mut self, action: &str) -> bool {
        let action = Self::canonical_key_action(action);
        match action {
            "interrupt" => {
                let sel_text = self.selection_text.borrow().clone();
                if self.selection_anchor.is_some() && !sel_text.is_empty() {
                    let copied = crate::image_paste::write_clipboard_text(&sel_text);
                    self.selection_anchor = None;
                    self.selection_focus = None;
                    *self.selection_text.borrow_mut() = String::new();
                    if copied {
                        self.notifications.push(
                            NotificationKind::Info,
                            "Copied to clipboard".to_string(),
                            Some(2),
                        );
                    }
                    return false;
                }

                if self.is_streaming {
                    self.is_streaming = false;
                    self.spinner_verb = None;
                    self.streaming_text.clear();
                    self.tool_use_blocks.clear();
                    self.status_message = Some("Cancelled.".to_string());
                } else {
                    self.should_quit = true;
                }
                false
            }
            "exit" => {
                if self.prompt_input.is_empty() {
                    self.should_quit = true;
                }
                false
            }
            "redraw" => false,
            "historySearch" => {
                let overlay = HistorySearchOverlay::open(&self.prompt_input.history);
                self.history_search_overlay = overlay;
                let mut hs = HistorySearch::new();
                hs.update_matches(&self.prompt_input.history);
                self.history_search = Some(hs);
                false
            }
            "openSearch" => {
                self.global_search.open();
                self.refresh_global_search();
                false
            }
            "globalSearch" => {
                self.global_search.open();
                self.refresh_global_search();
                false
            }
            "findInMessage" => {
                self.global_search.open();
                self.refresh_global_search();
                if self.transcript_search.total_matches > 0 {
                    let cur = self.transcript_search.current_match.unwrap_or(0);
                    self.status_message = Some(format!(
                        "Find: {}/{}",
                        cur + 1,
                        self.transcript_search.total_matches
                    ));
                }
                false
            }
            "findNext" => {
                self.transcript_search_step(true);
                false
            }
            "findPrev" => {
                self.transcript_search_step(false);
                false
            }
            "submit" => !self.is_streaming,
            "indent" => {
                if !self.is_streaming && !self.prompt_input.suggestions.is_empty() {
                    if self.prompt_input.suggestion_index.is_none() {
                        self.prompt_input.suggestion_index = Some(0);
                    }
                    self.prompt_input.accept_suggestion();
                    self.refresh_prompt_input();
                }
                false
            }
            "historyPrev" => {
                // Slash-command suggestions take priority over history.
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.suggestion_prev();
                    self.refresh_prompt_input();
                } else if !self.prompt_input.history.is_empty() {
                    self.prompt_input.history_up();
                    self.refresh_prompt_input();
                }
                false
            }
            "historyNext" => {
                // Slash-command suggestions take priority over history.
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.suggestion_next();
                    self.refresh_prompt_input();
                } else if self.prompt_input.history_pos.is_some() {
                    self.prompt_input.history_down();
                    self.refresh_prompt_input();
                }
                false
            }
            "scrollUp" => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
                self.auto_scroll = false;
                false
            }
            "scrollDown" => {
                let new_off = self.scroll_offset.saturating_sub(10);
                self.scroll_offset = new_off;
                if new_off == 0 {
                    self.auto_scroll = true;
                    self.new_messages_while_scrolled = 0;
                }
                false
            }
            "yes" => {
                self.permission_request = None;
                false
            }
            "no" => {
                self.permission_request = None;
                false
            }
            "prevOption" => {
                if let Some(pr) = self.permission_request.as_mut() {
                    if pr.selected_option > 0 {
                        pr.selected_option -= 1;
                    }
                }
                false
            }
            "nextOption" => {
                if let Some(pr) = self.permission_request.as_mut() {
                    if pr.selected_option + 1 < pr.options.len() {
                        pr.selected_option += 1;
                    }
                }
                false
            }
            "close" => {
                self.show_help = false;
                self.help_overlay.close();
                false
            }
            "select" => {
                // Legacy history search select
                if let Some(hs) = self.history_search.as_ref() {
                    if let Some(entry) = hs.current_entry(&self.prompt_input.history) {
                        self.set_prompt_text(entry.to_string());
                    }
                }
                self.history_search = None;
                self.history_search_overlay.close();
                false
            }
            "cancel" => {
                self.history_search = None;
                self.history_search_overlay.close();
                false
            }
            "prevResult" => {
                if let Some(hs) = self.history_search.as_mut() {
                    if hs.selected > 0 {
                        hs.selected -= 1;
                    }
                }
                self.history_search_overlay.select_prev();
                false
            }
            "nextResult" => {
                if let Some(hs) = self.history_search.as_mut() {
                    let max = hs.matches.len().saturating_sub(1);
                    if hs.selected < max {
                        hs.selected += 1;
                    }
                }
                self.history_search_overlay.select_next();
                false
            }
            // ========== NEW KEYBINDING ACTIONS (Phase 1) ==========
            "clearLine" => {
                // Ctrl+L: Clear the current input line (like bash Ctrl+L)
                self.prompt_input.text.clear();
                self.prompt_input.cursor = 0;
                self.refresh_prompt_input();
                false
            }
            "deleteCharBefore" => {
                // Ctrl+H: Delete character before cursor (backspace equivalent)
                if !self.is_streaming {
                    self.prompt_input.backspace();
                    self.refresh_prompt_input();
                }
                false
            }
            "previousMessage" => {
                // Alt+←: Navigate to previous message in transcript
                self.scroll_offset = self.scroll_offset.saturating_add(5);
                self.auto_scroll = false;
                false
            }
            "nextMessage" => {
                // Alt+→: Navigate to next message in transcript
                let new_off = self.scroll_offset.saturating_sub(5);
                self.scroll_offset = new_off;
                if new_off == 0 {
                    self.auto_scroll = true;
                }
                false
            }
            "jumpToNextError" => {
                // Ctrl+.: Jump to next error/issue in messages
                self.jump_to_next_error();
                false
            }
            "jumpToPreviousError" => {
                // Ctrl+Shift+.: Jump to previous error/issue in messages
                self.jump_to_previous_error();
                false
            }
            "reverseIndent" => {
                // Shift+Tab: Reverse indent (cycle permission mode)
                use mangocode_core::config::PermissionMode;
                self.config.permission_mode = match self.config.permission_mode {
                    PermissionMode::Default => PermissionMode::AcceptEdits,
                    PermissionMode::AcceptEdits => PermissionMode::BypassPermissions,
                    PermissionMode::BypassPermissions => PermissionMode::Default,
                    PermissionMode::Plan => PermissionMode::Default,
                };
                let label = match self.config.permission_mode {
                    PermissionMode::Default => "Default permissions",
                    PermissionMode::AcceptEdits => "Accept-edits mode",
                    PermissionMode::BypassPermissions => "Bypass permissions (dangerous)",
                    PermissionMode::Plan => "Plan mode",
                };
                self.status_message = Some(label.to_string());
                false
            }
            "openHelp" => {
                // Ctrl+H: Open help (alternative to F1)
                self.show_help = !self.show_help;
                self.help_overlay.toggle();
                false
            }
            "deleteWord" => {
                // Alt+D: Delete word forward
                if !self.is_streaming {
                    self.prompt_input.delete_word_at_cursor();
                    self.refresh_prompt_input();
                }
                false
            }
            "sendMessage" => {
                // Ctrl+M: Send message (alternative to Enter)
                // Keep slash-command suggestion behavior aligned with Enter.
                if !self.prompt_input.suggestions.is_empty()
                    && self.prompt_input.suggestion_index.is_some()
                    && self.prompt_input.text.starts_with('/')
                {
                    self.prompt_input.accept_suggestion();
                    self.refresh_prompt_input();
                    return false;
                }
                !self.is_streaming
            }
            _ => false,
        }
    }

    /// Handle a key event while in legacy history-search mode.
    fn handle_history_search_key(&mut self, key: KeyEvent) -> bool {
        let hs = match self.history_search.as_mut() {
            Some(h) => h,
            None => return false,
        };
        match key.code {
            KeyCode::Esc => {
                self.history_search = None;
                self.history_search_overlay.close();
            }
            KeyCode::Enter => {
                if let Some(entry) = hs.current_entry(&self.prompt_input.history) {
                    self.set_prompt_text(entry.to_string());
                }
                self.history_search = None;
                self.history_search_overlay.close();
            }
            KeyCode::Up => {
                if hs.selected > 0 {
                    hs.selected -= 1;
                }
            }
            KeyCode::Down => {
                let max = hs.matches.len().saturating_sub(1);
                if hs.selected < max {
                    hs.selected += 1;
                }
            }
            KeyCode::Backspace => {
                hs.query.pop();
                let history = self.prompt_input.history.clone();
                if let Some(hs) = self.history_search.as_mut() {
                    hs.update_matches(&history);
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                hs.query.push(c);
                let history = self.prompt_input.history.clone();
                if let Some(hs) = self.history_search.as_mut() {
                    hs.update_matches(&history);
                }
            }
            _ => {}
        }
        false
    }

    /// Handle a key event while a permission dialog is active.
    fn handle_permission_key(&mut self, key: KeyEvent) {
        let pr = match self.permission_request.as_mut() {
            Some(p) => p,
            None => return,
        };

        match key.code {
            KeyCode::Char(c) => {
                if let Some(digit) = c.to_digit(10) {
                    let idx = (digit as usize).saturating_sub(1);
                    if idx < pr.options.len() {
                        pr.selected_option = idx;
                    }
                } else {
                    // Check if any option matches this key.
                    let mut matched_idx = None;
                    for (i, opt) in pr.options.iter().enumerate() {
                        if opt.key == c {
                            matched_idx = Some(i);
                            break;
                        }
                    }
                    if let Some(idx) = matched_idx {
                        pr.selected_option = idx;
                        // If this is the prefix-allow option ('P'), record the prefix.
                        self.maybe_record_bash_prefix();
                        self.permission_request = None;
                    }
                }
            }
            KeyCode::Enter => {
                // If the currently selected option is the prefix-allow option, record it.
                self.maybe_record_bash_prefix();
                self.permission_request = None;
            }
            KeyCode::Up => {
                if let Some(pr) = self.permission_request.as_mut() {
                    if pr.selected_option > 0 {
                        pr.selected_option -= 1;
                    }
                }
            }
            KeyCode::Down => {
                if let Some(pr) = self.permission_request.as_mut() {
                    if pr.selected_option + 1 < pr.options.len() {
                        pr.selected_option += 1;
                    }
                }
            }
            KeyCode::Esc => {
                self.permission_request = None;
            }
            _ => {}
        }
    }

    /// If the active permission dialog's selected option is the prefix-allow
    /// option ('P') for a Bash dialog, extract the suggested prefix and add it
    /// to `bash_prefix_allowlist` so future requests with the same prefix are
    /// silently approved.
    fn maybe_record_bash_prefix(&mut self) {
        use crate::dialogs::PermissionDialogKind;
        let pr = match self.permission_request.as_ref() {
            Some(p) => p,
            None => return,
        };
        // Only act on Bash dialogs where the selected option key is 'P'.
        let selected_key = pr.options.get(pr.selected_option).map(|o| o.key);
        if selected_key != Some('P') {
            return;
        }
        if let PermissionDialogKind::Bash { command, .. } = &pr.kind {
            // Always normalize to the first whitespace-delimited word so
            // that the allowlist check in `bash_command_allowed_by_prefix`
            // (which also uses `split_whitespace().next()`) matches correctly.
            let first_word = command.split_whitespace().next().unwrap_or("").to_string();
            if !first_word.is_empty() {
                self.bash_prefix_allowlist.insert(first_word);
            }
        }
    }

    /// Returns `true` if the given bash `command` is covered by the session-local
    /// prefix allowlist (i.e. its first word matches an entry in
    /// `bash_prefix_allowlist`).  Used by callers to skip the permission dialog.
    pub fn bash_command_allowed_by_prefix(&self, command: &str) -> bool {
        let first_word = command.split_whitespace().next().unwrap_or("");
        !first_word.is_empty() && self.bash_prefix_allowlist.contains(first_word)
    }

    // ---- Advanced mouse interaction helpers --------------------------------

    /// Detect if a click is a double-click based on timing and position.
    /// Returns true if the click is within ~500ms and ~5px of the last click.
    fn is_double_click(&self, current_pos: (u16, u16)) -> bool {
        let now = std::time::Instant::now();
        match (self.last_click_time, self.last_click_position) {
            (Some(last_time), Some(last_pos)) => {
                let elapsed = now.duration_since(last_time);
                let distance = ((current_pos.0 as i32 - last_pos.0 as i32).abs()
                    + (current_pos.1 as i32 - last_pos.1 as i32).abs())
                    as u16;
                elapsed.as_millis() < 500 && distance <= 5
            }
            _ => false,
        }
    }

    /// Find word boundaries for the character at (col, row) in the selection text.
    /// Returns (start_col, end_col) for the word containing the given position.
    fn find_word_boundaries(&self, col: u16, _row: u16) -> Option<(u16, u16)> {
        // Get the current selection text to determine word boundaries
        let text = self.selection_text.borrow();
        if text.is_empty() {
            return None;
        }

        // For simplicity, we'll find the word based on whitespace and punctuation
        // In a full implementation, we'd map visual positions back to text offsets
        // For now, return a reasonable range around the click position
        let start = col.saturating_sub(10);
        let end = col.saturating_add(10);
        Some((start, end))
    }

    /// Find line boundaries for the row containing the click.
    /// Returns (start_row, end_row) for the line.
    fn find_line_boundaries(&self, row: u16) -> Option<(u16, u16)> {
        let selectable_area = self.last_selectable_area.get();
        let line_start = selectable_area.y;
        let line_end = selectable_area
            .y
            .saturating_add(selectable_area.height)
            .saturating_sub(1);

        if row >= line_start && row <= line_end {
            Some((row, row))
        } else {
            None
        }
    }

    fn context_menu_items(kind: ContextMenuKind) -> &'static [ContextMenuItem] {
        match kind {
            ContextMenuKind::Message { .. } => &[ContextMenuItem::Copy, ContextMenuItem::Fork],
            ContextMenuKind::Selection => &[ContextMenuItem::Copy],
        }
    }

    fn message_index_at_row(&self, row: u16) -> Option<usize> {
        self.message_row_map.borrow().get(&row).copied()
    }

    fn clear_selection(&mut self) {
        self.selection_anchor = None;
        self.selection_focus = None;
        *self.selection_text.borrow_mut() = String::new();
    }

    /// Show context menu at the given position.
    fn show_context_menu(&mut self, x: u16, y: u16, kind: ContextMenuKind) {
        self.context_menu_state = Some(ContextMenuState {
            x,
            y,
            selected_index: 0,
            kind,
        });
    }

    /// Dismiss the context menu.
    fn dismiss_context_menu(&mut self) {
        self.context_menu_state = None;
    }

    /// Handle context menu navigation with arrow keys.
    fn navigate_context_menu(&mut self, direction: KeyCode) {
        if let Some(mut menu) = self.context_menu_state {
            let item_count = Self::context_menu_items(menu.kind).len();
            match direction {
                KeyCode::Up => {
                    menu.selected_index = menu.selected_index.saturating_sub(1);
                }
                KeyCode::Down => {
                    menu.selected_index = (menu.selected_index + 1).min(item_count - 1);
                }
                _ => return,
            }
            self.context_menu_state = Some(menu);
        }
    }

    /// Execute the currently selected context menu item.
    fn execute_context_menu_item(&mut self) {
        if let Some(menu) = self.context_menu_state {
            let items = Self::context_menu_items(menu.kind);

            if menu.selected_index < items.len() {
                let item = items[menu.selected_index];
                self.handle_context_menu_action(item, menu.kind);
            }
        }
        self.dismiss_context_menu();
    }

    /// Handle a context menu action.
    fn handle_context_menu_action(&mut self, item: ContextMenuItem, kind: ContextMenuKind) {
        match item {
            ContextMenuItem::Copy => {
                let text = match kind {
                    ContextMenuKind::Message { message_index } => self
                        .messages
                        .get(message_index)
                        .map(|message| message.get_all_text()),
                    ContextMenuKind::Selection => {
                        let selected = self.selection_text.borrow().trim().to_string();
                        if selected.is_empty() {
                            None
                        } else {
                            Some(selected)
                        }
                    }
                };

                if let Some(text) = text {
                    if crate::message_copy::copy_to_clipboard(&text) {
                        self.notifications.push(
                            NotificationKind::Info,
                            format!("Copied {} chars to clipboard.", text.len()),
                            Some(3),
                        );
                    } else {
                        self.notifications.push(
                            NotificationKind::Warning,
                            "Failed to copy to clipboard.".to_string(),
                            Some(3),
                        );
                    }
                    debug!("Copy action triggered, text: {} chars", text.len());
                }
            }
            ContextMenuItem::Fork => {
                if let ContextMenuKind::Message { message_index } = kind {
                    let branch_point = message_index + 1;
                    self.prompt_input
                        .replace_text(format!("/fork {}", branch_point));
                    self.status_message = Some(format!(
                        "Fork at message {} - press Enter to confirm",
                        branch_point
                    ));
                }
            }
        }
    }

    /// Process mouse events (trackpad scroll, text selection, etc.).
    pub fn handle_mouse_event(&mut self, mouse_event: MouseEvent) {
        use crossterm::event::MouseButton;

        if matches!(mouse_event.kind, MouseEventKind::Moved) {
            let msg_area = self.last_msg_area.get();
            let in_msg_area = mouse_event.column >= msg_area.x
                && mouse_event.column < msg_area.x.saturating_add(msg_area.width)
                && mouse_event.row >= msg_area.y
                && mouse_event.row < msg_area.y.saturating_add(msg_area.height);
            self.hovered_message_index = if in_msg_area {
                self.message_index_at_row(mouse_event.row)
            } else {
                None
            };
            return;
        }

        // ---- Dialog interaction: dismiss on click-outside, scroll/click inside ----
        // Key-input and device-auth stay outside this gate so their visible text
        // can still be selected and copied with the mouse.
        let any_dialog = self.connect_dialog.visible
            || self.command_palette.visible
            || self.model_picker.visible
            || self.export_dialog.visible
            || self.settings_screen.visible
            || self.stats_dialog.open
            || self.context_viz.visible
            || self.session_browser.visible;

        if any_dialog {
            match mouse_event.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    // DialogSelect dialogs — check if click is inside for item selection
                    let in_dialog = if self.connect_dialog.visible {
                        self.connect_dialog
                            .contains(mouse_event.column, mouse_event.row)
                    } else if self.command_palette.visible {
                        self.command_palette
                            .contains(mouse_event.column, mouse_event.row)
                    } else {
                        // Other dialogs (model_picker, settings, export, etc.) —
                        // treat any click as "inside" to prevent accidental dismiss.
                        // User must press Esc to close these.
                        true
                    };

                    if in_dialog {
                        // Click inside a DialogSelect — select the clicked item
                        if self.connect_dialog.visible {
                            self.connect_dialog.handle_mouse_click(mouse_event.row);
                        } else if self.command_palette.visible {
                            self.command_palette.handle_mouse_click(mouse_event.row);
                        }
                        // Other dialogs: click absorbed, no action needed
                    } else {
                        // Click outside a DialogSelect — dismiss and restore input focus
                        self.close_secondary_views();
                        self.focus = FocusTarget::Input;
                    }
                }
                MouseEventKind::ScrollUp => {
                    // Scroll through dialog items
                    if self.connect_dialog.visible {
                        self.connect_dialog.move_up();
                    } else if self.command_palette.visible {
                        self.command_palette.move_up();
                    }
                }
                MouseEventKind::ScrollDown => {
                    if self.connect_dialog.visible {
                        self.connect_dialog.move_down();
                    } else if self.command_palette.visible {
                        self.command_palette.move_down();
                    }
                }
                _ => {}
            }
            return; // Don't process any other mouse events when a dialog is open
        }

        match mouse_event.kind {
            MouseEventKind::ScrollUp => {
                // Don't consume Ctrl+Scroll — let the terminal handle zoom.
                if !mouse_event.modifiers.contains(KeyModifiers::CONTROL) {
                    let step = self.scroll_step();
                    self.scroll_offset = self.scroll_offset.saturating_add(step);
                    self.auto_scroll = false;
                }
            }
            MouseEventKind::ScrollDown => {
                if !mouse_event.modifiers.contains(KeyModifiers::CONTROL) {
                    let step = self.scroll_step();
                    let new_off = self.scroll_offset.saturating_sub(step);
                    self.scroll_offset = new_off;
                    if new_off == 0 {
                        self.auto_scroll = true;
                        self.new_messages_while_scrolled = 0;
                    }
                }
            }
            // ---- Right-click context menu ----------------------------------
            MouseEventKind::Down(MouseButton::Right) => {
                let msg_area = self.last_msg_area.get();
                let has_selection = !self.selection_text.borrow().trim().is_empty();
                if mouse_event.column >= msg_area.x
                    && mouse_event.column < msg_area.x.saturating_add(msg_area.width)
                    && mouse_event.row >= msg_area.y
                    && mouse_event.row < msg_area.y.saturating_add(msg_area.height)
                {
                    if let Some(message_index) = self.message_index_at_row(mouse_event.row) {
                        self.show_context_menu(
                            mouse_event.column,
                            mouse_event.row,
                            ContextMenuKind::Message { message_index },
                        );
                    } else {
                        self.dismiss_context_menu();
                    }
                } else if has_selection {
                    self.show_context_menu(
                        mouse_event.column,
                        mouse_event.row,
                        ContextMenuKind::Selection,
                    );
                } else {
                    self.dismiss_context_menu();
                }
            }

            // ---- Text selection / focus routing -------------------------
            MouseEventKind::Down(MouseButton::Left) => {
                // Dismiss context menu on left click
                self.dismiss_context_menu();

                let input_area = self.last_input_area.get();
                let selectable_area = self.last_selectable_area.get();

                let in_input = input_area.width > 0
                    && input_area.height > 0
                    && mouse_event.row >= input_area.y
                    && mouse_event.row < input_area.y.saturating_add(input_area.height)
                    && mouse_event.column >= input_area.x
                    && mouse_event.column < input_area.x.saturating_add(input_area.width);

                let in_selectable = selectable_area.width > 0
                    && selectable_area.height > 0
                    && mouse_event.row >= selectable_area.y
                    && mouse_event.row < selectable_area.y.saturating_add(selectable_area.height)
                    && mouse_event.column >= selectable_area.x
                    && mouse_event.column < selectable_area.x.saturating_add(selectable_area.width);

                if in_input {
                    self.focus = FocusTarget::Input;
                    self.clear_selection();
                } else if selectable_area.width == 0 || selectable_area.height == 0 {
                    self.click_count = 0;
                } else if in_selectable {
                    self.focus = FocusTarget::Transcript;

                    let current_pos = (mouse_event.column, mouse_event.row);
                    let now = std::time::Instant::now();

                    if mouse_event.modifiers.contains(KeyModifiers::ALT) {
                        if let Some((start_row, end_row)) = self.find_line_boundaries(current_pos.1) {
                            self.selection_anchor = Some((selectable_area.x, start_row));
                            self.selection_focus = Some((
                                selectable_area
                                    .x
                                    .saturating_add(selectable_area.width)
                                    .saturating_sub(1),
                                end_row,
                            ));
                        }
                        self.last_click_time = Some(now);
                        self.last_click_position = Some(current_pos);
                        self.click_count = 0;
                        return;
                    }

                    // Check for double-click
                    if self.is_double_click(current_pos) {
                        self.click_count += 1;
                        if self.click_count >= 3 {
                            // Triple-click: select entire line
                            self.selection_anchor = Some((selectable_area.x, current_pos.1));
                            self.selection_focus = Some((
                                selectable_area
                                    .x
                                    .saturating_add(selectable_area.width)
                                    .saturating_sub(1),
                                current_pos.1,
                            ));
                            self.click_count = 0; // Reset for next click sequence
                        } else {
                            // Double-click: select word
                            if let Some((start, end)) =
                                self.find_word_boundaries(current_pos.0, current_pos.1)
                            {
                                self.selection_anchor = Some((start, current_pos.1));
                                self.selection_focus = Some((end, current_pos.1));
                            }
                        }
                    } else {
                        // Single click or new click sequence
                        self.click_count = 1;
                        self.selection_anchor = Some(current_pos);
                        self.selection_focus = Some(current_pos);
                        *self.selection_text.borrow_mut() = String::new();
                    }

                    self.last_click_time = Some(now);
                    self.last_click_position = Some(current_pos);
                } else {
                    self.click_count = 0;
                    self.clear_selection();
                }
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                // Dismiss context menu on drag
                self.dismiss_context_menu();

                // Continue drag — clamp to the selectable frame bounds so dragging
                // outside extends selection to the edge rather than cancelling.
                if self.selection_anchor.is_some() {
                    let selectable_area = self.last_selectable_area.get();
                    if selectable_area.width > 0 && selectable_area.height > 0 {
                        let clamped_col = mouse_event.column.max(selectable_area.x).min(
                            selectable_area
                                .x
                                .saturating_add(selectable_area.width)
                                .saturating_sub(1),
                        );
                        let clamped_row = mouse_event.row.max(selectable_area.y).min(
                            selectable_area
                                .y
                                .saturating_add(selectable_area.height)
                                .saturating_sub(1),
                        );
                        self.selection_focus = Some((clamped_col, clamped_row));
                        self.click_count = 0; // Reset on drag to prevent further double-clicks
                    }
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                // Clear if no actual drag (single click = no selection)
                if self.selection_anchor == self.selection_focus {
                    self.clear_selection();
                }
            }
            _ => {}
        }
    }

    // -------------------------------------------------------------------
    // Query event handling
    // -------------------------------------------------------------------

    /// Push a completed assistant message and trigger auto-scroll bookkeeping.
    fn push_assistant_message(&mut self, text: String) {
        let msg = Message::assistant(text);
        self.messages.push(msg);
        self.invalidate_transcript();
        self.on_new_message();
    }

    /// Process a query event from the agentic loop.
    pub fn handle_query_event(&mut self, event: QueryEvent) {
        match event {
            QueryEvent::Stream(stream_evt)
            | QueryEvent::StreamWithParent {
                event: stream_evt, ..
            } => {
                if !self.is_streaming {
                    let seed = self.frame_count as usize ^ (self.messages.len() * 17);
                    self.spinner_verb = Some(sample_spinner_verb(seed).to_string());
                    // Fallback only: if a user message was not pushed first,
                    // start timing when the first stream event arrives.
                    if self.turn_start.is_none() {
                        self.turn_start = Some(std::time::Instant::now());
                        self.last_turn_elapsed = None;
                        self.last_turn_verb = None;
                    }
                }
                self.is_streaming = true;
                match stream_evt {
                    mangocode_api::AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                        // Reset stall timer on any incoming delta — we're making progress.
                        self.stall_start = None;
                        match delta {
                            mangocode_api::streaming::ContentDelta::TextDelta { text } => {
                                self.streaming_text.push_str(&text);
                                self.invalidate_transcript();
                            }
                            mangocode_api::streaming::ContentDelta::ThinkingDelta { thinking } => {
                                debug!(len = thinking.len(), "Thinking delta received");
                            }
                            _ => {}
                        }
                    }
                    mangocode_api::AnthropicStreamEvent::MessageStop => {
                        self.is_streaming = false;
                        self.spinner_verb = None;
                        self.stall_start = None;
                        if !self.streaming_text.is_empty() {
                            let text = std::mem::take(&mut self.streaming_text);
                            self.push_assistant_message(text);
                        }
                    }
                    _ => {
                        // Any other stream event: if we have no stall_start yet,
                        // record now so the red-spinner timer can begin.
                        if self.stall_start.is_none() {
                            self.stall_start = Some(std::time::Instant::now());
                        }
                    }
                }
            }

            QueryEvent::ToolStart {
                tool_name,
                tool_id,
                input_json,
                ..
            } => {
                if !self.is_streaming && self.spinner_verb.is_none() {
                    let seed = self.frame_count as usize ^ (self.messages.len() * 17);
                    self.spinner_verb = Some(sample_spinner_verb(seed).to_string());
                }
                self.is_streaming = true;
                self.status_message = Some(format!("Running {}…", tool_name));
                if let Some(existing) = self.tool_use_blocks.iter_mut().find(|b| b.id == tool_id) {
                    existing.status = ToolStatus::Running;
                    existing.output_preview = None;
                    existing.input_json = input_json;
                } else {
                    self.tool_use_blocks.push(ToolUseBlock {
                        id: tool_id,
                        name: tool_name,
                        status: ToolStatus::Running,
                        output_preview: None,
                        input_json,
                    });
                }
                self.invalidate_transcript();
            }

            QueryEvent::ToolEnd {
                tool_name: _,
                tool_id,
                result,
                is_error,
                ..
            } => {
                // Build a multi-line preview: show up to 3 lines, truncate if more.
                let all_lines: Vec<&str> = result.lines().collect();
                let preview_lines = all_lines.len().min(3);
                let mut preview = all_lines[..preview_lines].join("\n");
                let remaining = all_lines.len().saturating_sub(preview_lines);
                if remaining > 0 {
                    preview.push_str(&format!("\n\u{2026} {} more lines", remaining));
                }
                if let Some(block) = self.tool_use_blocks.iter_mut().find(|b| b.id == tool_id) {
                    block.status = if is_error {
                        ToolStatus::Error
                    } else {
                        ToolStatus::Done
                    };
                    block.output_preview = Some(preview);
                }
                self.invalidate_transcript();
                if is_error {
                    self.status_message = Some(format!("Tool error: {}", result));
                } else {
                    self.status_message = None;
                }
                self.refresh_turn_diff_from_history();
            }

            QueryEvent::TurnComplete {
                turn, stop_reason, ..
            } => {
                debug!(turn, stop_reason, "Turn complete");
                self.is_streaming = false;
                self.spinner_verb = None;
                self.status_message = None;
                self.stall_start = None;
                // Record elapsed time and pick a completion verb
                if let Some(start) = self.turn_start.take() {
                    let secs = start.elapsed().as_secs();
                    let seed = self.frame_count as usize ^ (self.messages.len() * 7);
                    self.last_turn_elapsed = Some(format_elapsed(secs));
                    self.last_turn_verb = Some(sample_completion_verb(seed));
                }
                if !self.streaming_text.is_empty() {
                    let text = std::mem::take(&mut self.streaming_text);
                    self.push_assistant_message(text);
                }
                // Clear transient tool blocks — the canonical assistant message
                // (with inline ToolUse / ToolResult content blocks) is synced into
                // `app.messages` from the query task at task-finished time, so the
                // bottom-of-screen staging area must not retain completed entries.
                self.tool_use_blocks.clear();
                self.invalidate_transcript();
                self.refresh_turn_diff_from_history();
            }

            QueryEvent::Status(msg) => {
                self.status_message = Some(msg);
            }

            QueryEvent::Error(msg) => {
                self.is_streaming = false;
                self.spinner_verb = None;
                self.streaming_text.clear();
                self.invalidate_transcript();
                let err_msg = format!("Error: {}", msg);
                self.push_assistant_message(err_msg.clone());
                self.status_message = Some(err_msg);
            }
            QueryEvent::TokenWarning { state, pct_used } => {
                // Push a notification for context window warnings (notification + threshold tracking).
                use mangocode_query::compact::TokenWarningState;

                // Only escalate — never repeat a threshold already shown.
                match state {
                    TokenWarningState::Ok => {
                        // Reset threshold tracking when back to normal
                        self.token_warning_threshold_shown = 0;
                    }
                    TokenWarningState::Warning if self.token_warning_threshold_shown < 80 => {
                        self.token_warning_threshold_shown = 80;
                        self.notifications.push(
                            NotificationKind::Warning,
                            format!(
                                "Context window {:.0}% full. Consider /compact.",
                                pct_used * 100.0
                            ),
                            Some(30),
                        );
                    }
                    TokenWarningState::Critical if self.token_warning_threshold_shown < 95 => {
                        self.token_warning_threshold_shown = 95;
                        self.notifications.push(
                            NotificationKind::Error,
                            format!(
                                "Context window {:.0}% full! Run /compact now.",
                                pct_used * 100.0
                            ),
                            None,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Update token count from tracker.
        self.token_count = self.cost_tracker.total_tokens() as u32;
    }

    // -------------------------------------------------------------------
    // Main run loop
    // -------------------------------------------------------------------

    /// Run the TUI event loop. Returns `Some(input)` when the user submits
    /// a message, or `None` when the user quits.
    pub fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> anyhow::Result<Option<String>> {
        loop {
            let frame_loop_start = std::time::Instant::now();
            self.frame_count = self.frame_count.wrapping_add(1);

            // Sync cost/token counters from the shared tracker
            self.cost_usd = self.cost_tracker.total_cost_usd();
            self.token_count = self.cost_tracker.total_tokens() as u32;

            // Expire old notifications
            self.notifications.tick();
            self.memory_update_notification.tick();

            // Drain background model-fetch results (non-blocking).
            if let Some(ref mut rx) = self.model_fetch_rx {
                match rx.try_recv() {
                    Ok(Ok(entries)) => {
                        let provider = self.config.provider.as_deref().unwrap_or("anthropic");
                        let provider_prefix = format!("{}/", provider);
                        let current = self
                            .model_name
                            .strip_prefix(&provider_prefix)
                            .unwrap_or(self.model_name.as_str())
                            .to_string();
                        self.model_picker.set_models(entries);
                        // Re-apply the current-model highlight so it stays accurate.
                        for m in &mut self.model_picker.models {
                            m.is_current = m.id == current;
                        }
                        self.model_fetch_rx = None;
                    }
                    Ok(Err(())) | Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        self.model_picker.loading_models = false;
                        self.model_fetch_rx = None;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
                }
            }

            // Spawn async provider model-list fetch when requested.
            if self.model_picker_fetch_pending {
                self.model_picker_fetch_pending = false;
                let provider_id_str = self
                    .config
                    .provider
                    .clone()
                    .unwrap_or_else(|| "anthropic".to_string());
                if let Some(ref registry) = self.provider_registry {
                    let pid = mangocode_core::ProviderId::new(&provider_id_str);
                    if let Some(provider) = registry.get(&pid) {
                        let provider = provider.clone();
                        let (tx, rx) = tokio::sync::mpsc::channel(1);
                        self.model_fetch_rx = Some(rx);
                        self.model_picker.loading_models = true;
                        tokio::spawn(async move {
                            match provider.list_models().await {
                                Ok(models) => {
                                    let entries: Vec<crate::model_picker::ModelEntry> = models
                                        .into_iter()
                                        .map(|m| {
                                            let ctx_k = m.context_window / 1000;
                                            crate::model_picker::ModelEntry {
                                                id: m.id.to_string(),
                                                display_name: m.name.clone(),
                                                description: format!("{}K context", ctx_k),
                                                is_current: false,
                                            }
                                        })
                                        .collect();
                                    let _ = tx.send(Ok(entries)).await;
                                }
                                Err(_) => {
                                    let _ = tx.send(Err(())).await;
                                }
                            }
                        });
                    }
                }
            }

            // Drain voice transcription events (non-blocking).
            // When the background recording/transcription task emits a
            // TranscriptReady event we insert the text directly into the
            // prompt so the user can review and submit it.
            {
                use mangocode_core::voice::VoiceEvent;
                let mut events = Vec::new();
                if let Some(ref mut rx) = self.voice_event_rx {
                    while let Ok(ev) = rx.try_recv() {
                        events.push(ev);
                    }
                }
                for ev in events {
                    match ev {
                        VoiceEvent::RecordingStarted => {
                            self.voice_recording = true;
                            self.status_message = Some(
                                "Recording\u{2026} press V again or Enter to stop".to_string(),
                            );
                        }
                        VoiceEvent::RecordingStopped => {
                            self.voice_recording = false;
                            self.status_message = Some("Transcribing\u{2026}".to_string());
                        }
                        VoiceEvent::TranscriptReady(text) => {
                            if !text.is_empty() {
                                // Append to existing prompt text with a space separator
                                // so the user can combine voice + typed input.
                                if !self.prompt_input.text.is_empty()
                                    && !self.prompt_input.text.ends_with(' ')
                                {
                                    self.prompt_input.paste(" ");
                                }
                                self.prompt_input.paste(&text);
                                self.refresh_prompt_input();
                                self.status_message =
                                    Some(format!("Transcribed: {}", &text[..text.len().min(60)]));
                            }
                            // Clear the channel once we have the result.
                            self.voice_event_rx = None;
                        }
                        VoiceEvent::Error(msg) => {
                            self.voice_recording = false;
                            self.voice_event_rx = None;
                            self.notifications.push(
                                NotificationKind::Warning,
                                format!("Voice: {}", msg),
                                Some(8),
                            );
                        }
                    }
                }
            }

            // Refresh task list if the overlay is visible (every frame for live updates)
            if self.tasks_overlay.visible {
                self.tasks_overlay
                    .refresh_tasks(&mangocode_tools::TASK_STORE);
            }

            // Draw the frame
            let render_start = std::time::Instant::now();
            terminal.draw(|f| render::render_app(f, self))?;
            self.stats_dialog
                .update_perf_metrics(None, Some(render_start.elapsed().as_micros() as u64), None);

            // Poll for events with a short timeout so we can redraw for animation
            if event::poll(std::time::Duration::from_millis(50))? {
                let input_start = std::time::Instant::now();
                match event::read()? {
                    Event::Key(key) => {
                        // On Windows crossterm fires both Press and Release events.
                        // We normally skip non-press events, but when voice PTT mode
                        // is active we need the Release event for the `V` key so we
                        // can stop recording as soon as the user lifts the key.
                        if key.kind != crossterm::event::KeyEventKind::Press {
                            // Handle V-key release to stop PTT recording.
                            if key.kind == crossterm::event::KeyEventKind::Release
                                && key.code == KeyCode::Char('v')
                                && key.modifiers == KeyModifiers::NONE
                                && self.voice_recording
                                && self.voice_recorder.is_some()
                            {
                                self.handle_voice_ptt_stop();
                            }
                            continue;
                        }
                        let should_submit = self.handle_key_event(key);
                        // Honour `:q`/`:wq` from vim command-line mode
                        if self.prompt_input.vim_quit_requested {
                            self.prompt_input.vim_quit_requested = false;
                            self.should_quit = true;
                        }
                        if self.should_quit {
                            return Ok(None);
                        }
                        if should_submit {
                            // Check if this is a slash command that should open a UI screen
                            if crate::input::is_slash_command(&self.prompt_input.text) {
                                let cmd = {
                                    let (c, _) =
                                        crate::input::parse_slash_command(&self.prompt_input.text);
                                    c.to_string()
                                };
                                if self.intercept_slash_command(&cmd) {
                                    self.clear_prompt();
                                    continue;
                                }
                            }
                            let input = self.take_input();
                            if !input.is_empty() {
                                return Ok(Some(input));
                            }
                        }
                    }
                    Event::Paste(data)
                        if !self.is_streaming
                            && self.permission_request.is_none()
                            && !self.history_search_overlay.visible
                            && self.history_search.is_none() =>
                    {
                        self.handle_prompt_paste(&data, std::time::Instant::now());
                    }
                    Event::Mouse(mouse_event) => {
                        self.handle_mouse_event(mouse_event);
                    }
                    _ => {}
                }
                self.stats_dialog
                    .update_perf_metrics(Some(input_start.elapsed().as_micros() as u64), None, None);
            }

            self.stats_dialog.update_perf_metrics(
                None,
                None,
                Some(frame_loop_start.elapsed().as_secs_f64() * 1000.0),
            );
        }
    }

    // ========== NEW KEYBINDING HELPER FUNCTIONS (Phase 1) ==========

    /// Jump to the next error/issue in messages.
    /// Searches for common error indicators: "Error:", "ERROR:", "error", "failed", "FAIL".
    fn jump_to_next_error(&mut self) {
        const ERROR_KEYWORDS: &[&str] = &["error:", "failed:", "fail"];

        // Search forward from current position
        for i in 0..self.messages.len() {
            let msg = &self.messages[i];
            let content = msg.get_all_text().to_lowercase();

            // Check if message contains error keywords
            let has_error = ERROR_KEYWORDS
                .iter()
                .any(|keyword| content.contains(keyword));

            if has_error && i > (self.messages.len().saturating_sub(self.scroll_offset / 2)) {
                // Found an error message, scroll to it
                let new_offset = self.messages.len().saturating_sub(i);
                self.scroll_offset = new_offset.saturating_mul(2);
                self.auto_scroll = false;
                self.status_message = Some(format!("Error found in message {}", i + 1));
                return;
            }
        }

        self.status_message = Some("No more errors found.".to_string());
    }

    /// Jump to the previous error/issue in messages.
    /// Searches backwards for common error indicators.
    fn jump_to_previous_error(&mut self) {
        const ERROR_KEYWORDS: &[&str] = &["error:", "failed:", "fail"];

        // Search backward from current position
        for i in (0..self.messages.len()).rev() {
            let msg = &self.messages[i];
            let content = msg.get_all_text().to_lowercase();

            // Check if message contains error keywords
            let has_error = ERROR_KEYWORDS
                .iter()
                .any(|keyword| content.contains(keyword));

            if has_error && i < (self.messages.len().saturating_sub(self.scroll_offset / 2)) {
                // Found an error message, scroll to it
                let new_offset = self.messages.len().saturating_sub(i);
                self.scroll_offset = new_offset.saturating_mul(2);
                self.auto_scroll = false;
                self.status_message = Some(format!("Error found in message {}", i + 1));
                return;
            }
        }

        self.status_message = Some("No previous errors found.".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    fn make_app() -> App {
        let config = Config::default();
        let cost_tracker = mangocode_core::cost::CostTracker::new();
        App::new(config, cost_tracker)
    }

    fn key_press(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn test_clear_slash_command_clears_messages() {
        let mut app = make_app();
        app.add_message(Role::User, "hello".to_string());
        app.add_message(Role::Assistant, "world".to_string());
        assert_eq!(app.messages.len(), 2);
        assert!(app.intercept_slash_command("clear"));
        assert_eq!(app.messages.len(), 0);
    }

    #[test]
    fn test_exit_slash_command_sets_quit_flag() {
        let mut app = make_app();
        assert!(!app.should_quit);
        assert!(app.intercept_slash_command("exit"));
        assert!(app.should_quit);
    }

    #[test]
    fn test_vim_slash_command_toggles_vim() {
        let mut app = make_app();
        assert!(!app.prompt_input.vim_enabled);
        assert!(app.intercept_slash_command("vim"));
        assert!(app.prompt_input.vim_enabled);
        assert!(app.intercept_slash_command("vim"));
        assert!(!app.prompt_input.vim_enabled);
    }

    #[test]
    fn test_model_slash_command_opens_picker() {
        let mut app = make_app();
        assert!(!app.model_picker.visible);
        assert!(app.intercept_slash_command("model"));
        assert!(app.model_picker.visible);
    }

    #[test]
    fn test_fast_slash_command_toggles_fast_mode() {
        let mut app = make_app();
        assert!(!app.fast_mode);
        assert!(app.intercept_slash_command("fast"));
        assert!(app.fast_mode);
        assert!(app.intercept_slash_command("fast"));
        assert!(!app.fast_mode);
    }

    #[test]
    fn test_output_style_cycles() {
        let mut app = make_app();
        assert_eq!(app.output_style, "auto");
        assert!(app.intercept_slash_command("output-style"));
        assert_eq!(app.output_style, "stream");
        assert!(app.intercept_slash_command("output-style"));
        assert_eq!(app.output_style, "verbose");
        assert!(app.intercept_slash_command("output-style"));
        assert_eq!(app.output_style, "auto");
    }

    #[test]
    fn test_context_menu_fork_targets_clicked_message() {
        let mut app = make_app();
        app.add_message(Role::User, "one".to_string());
        app.add_message(Role::Assistant, "two".to_string());
        app.add_message(Role::User, "three".to_string());

        app.handle_context_menu_action(
            ContextMenuItem::Fork,
            ContextMenuKind::Message { message_index: 1 },
        );

        assert_eq!(app.prompt_input.text, "/fork 2");
        assert_eq!(
            app.status_message.as_deref(),
            Some("Fork at message 2 - press Enter to confirm")
        );
    }

    #[test]
    fn test_right_click_targets_row_message_instead_of_last_message() {
        use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

        let mut app = make_app();
        app.last_msg_area.set(ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 10,
        });
        app.message_row_map.borrow_mut().insert(3, 1);

        app.handle_mouse_event(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Right),
            column: 12,
            row: 3,
            modifiers: KeyModifiers::empty(),
        });

        assert!(matches!(
            app.context_menu_state,
            Some(ContextMenuState {
                kind: ContextMenuKind::Message { message_index: 1 },
                ..
            })
        ));
    }

    #[test]
    fn test_mouse_hover_sets_and_clears_hovered_message_index() {
        use crossterm::event::{KeyModifiers, MouseEvent, MouseEventKind};

        let mut app = make_app();
        app.last_msg_area.set(ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 10,
        });
        app.message_row_map.borrow_mut().insert(4, 7);

        app.handle_mouse_event(MouseEvent {
            kind: MouseEventKind::Moved,
            column: 10,
            row: 4,
            modifiers: KeyModifiers::empty(),
        });
        assert_eq!(app.hovered_message_index, Some(7));

        app.handle_mouse_event(MouseEvent {
            kind: MouseEventKind::Moved,
            column: 120,
            row: 40,
            modifiers: KeyModifiers::empty(),
        });
        assert_eq!(app.hovered_message_index, None);
    }

    // ---- Help overlay -------------------------------------------------------

    #[test]
    fn test_help_slash_command_opens_overlay() {
        let mut app = make_app();
        assert!(!app.help_overlay.visible);
        assert!(!app.show_help);
        assert!(app.intercept_slash_command("help"));
        assert!(app.help_overlay.visible);
        assert!(app.show_help);
    }

    #[test]
    fn test_help_slash_command_is_idempotent_when_already_open() {
        let mut app = make_app();
        // First call opens it.
        assert!(app.intercept_slash_command("help"));
        assert!(app.help_overlay.visible);
        // Second call while already open should leave it open (not toggle it off).
        assert!(app.intercept_slash_command("help"));
        assert!(app.help_overlay.visible);
    }

    #[test]
    fn test_question_mark_in_prompt_inserts_literal_character() {
        let mut app = make_app();
        app.set_prompt_text("hello".to_string());

        let _ = app.handle_key_event(key_press(KeyCode::Char('?'), KeyModifiers::empty()));

        assert_eq!(app.prompt_input.text, "hello?");
        assert!(!app.help_overlay.visible);
        assert!(!app.show_help);
    }

    // ---- Bash prefix allowlist ----------------------------------------------

    #[test]
    fn test_bash_command_not_allowed_by_default() {
        let app = make_app();
        assert!(!app.bash_command_allowed_by_prefix("git status"));
        assert!(!app.bash_command_allowed_by_prefix("ls -la"));
        assert!(!app.bash_command_allowed_by_prefix(""));
    }

    #[test]
    fn test_bash_prefix_allowlist_after_p_key() {
        use crate::dialogs::PermissionRequest;
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

        let mut app = make_app();
        // Set up a bash permission dialog with a suggested prefix.
        let pr = PermissionRequest::bash(
            "tu-1".to_string(),
            "Bash".to_string(),
            "wants to run".to_string(),
            "git status".to_string(),
            Some("git".to_string()),
        );
        app.permission_request = Some(pr);

        // Simulate pressing 'P' (prefix-allow key).
        let key = KeyEvent {
            code: KeyCode::Char('P'),
            modifiers: KeyModifiers::SHIFT,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        app.handle_permission_key(key);

        // Dialog should be dismissed and "git" added to the allowlist.
        assert!(app.permission_request.is_none());
        assert!(app.bash_command_allowed_by_prefix("git status"));
        assert!(app.bash_command_allowed_by_prefix("git push origin main"));
        // Other commands should NOT be allowed.
        assert!(!app.bash_command_allowed_by_prefix("rm -rf /tmp"));
    }

    #[test]
    fn test_bash_prefix_allowlist_via_enter_on_p_option() {
        use crate::dialogs::PermissionRequest;
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

        let mut app = make_app();
        let mut pr = PermissionRequest::bash(
            "tu-2".to_string(),
            "Bash".to_string(),
            "wants to run".to_string(),
            "cargo build".to_string(),
            Some("cargo".to_string()),
        );
        // Navigate to the prefix option (index 3 in a 5-option dialog).
        pr.selected_option = 3;
        app.permission_request = Some(pr);

        // Press Enter to confirm the currently selected (prefix) option.
        let key = KeyEvent {
            code: KeyCode::Enter,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        app.handle_permission_key(key);

        assert!(app.permission_request.is_none());
        assert!(app.bash_command_allowed_by_prefix("cargo test"));
        assert!(!app.bash_command_allowed_by_prefix("make build"));
    }

    #[test]
    fn test_bash_prefix_allowlist_non_prefix_option_does_not_add() {
        use crate::dialogs::PermissionRequest;
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

        let mut app = make_app();
        let pr = PermissionRequest::bash(
            "tu-3".to_string(),
            "Bash".to_string(),
            "wants to run".to_string(),
            "npm install".to_string(),
            Some("npm".to_string()),
        );
        app.permission_request = Some(pr);

        // Press 'y' (allow-once) — should NOT add to allowlist.
        let key = KeyEvent {
            code: KeyCode::Char('y'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        app.handle_permission_key(key);

        assert!(app.permission_request.is_none());
        assert!(!app.bash_command_allowed_by_prefix("npm test"));
    }

    #[test]
    fn test_bracketed_multiline_paste_does_not_auto_submit() {
        let mut app = make_app();
        app.handle_prompt_paste("line 1\nline 2\nline 3", std::time::Instant::now());

        let submit = app.handle_key_event(key_press(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!submit);
        // Multi-line paste is collapsed into a placeholder.
        assert_eq!(app.prompt_input.text, "[Pasted text #1 +3 lines]\n");
    }

    #[test]
    fn test_rapid_printable_stream_with_enter_inserts_newline_not_submit() {
        let mut app = make_app();
        for ch in ['a', 'b', 'c', 'd', 'e'] {
            let _ = app.handle_key_event(key_press(KeyCode::Char(ch), KeyModifiers::NONE));
        }

        let submit = app.handle_key_event(key_press(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!submit);
        assert_eq!(app.prompt_input.text, "abcde\n");
    }

    #[test]
    fn test_normal_typing_then_enter_still_submits() {
        let mut app = make_app();
        let _ = app.handle_key_event(key_press(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = app.handle_key_event(key_press(KeyCode::Char('i'), KeyModifiers::NONE));

        let submit = app.handle_key_event(key_press(KeyCode::Enter, KeyModifiers::NONE));
        assert!(submit);
        assert_eq!(app.prompt_input.text, "hi");
    }

    #[test]
    fn test_transcript_search_counts_multiline_matches_case_insensitive() {
        let mut app = make_app();
        app.add_message(Role::User, "Hello\nworld hello".to_string());
        app.add_message(Role::Assistant, "HELLO again".to_string());
        app.transcript_search.query = "hello".to_string();
        app.recompute_transcript_search();

        assert_eq!(app.transcript_search.total_matches, 3);
        assert_eq!(app.transcript_search.current_match, Some(0));
    }

    #[test]
    fn test_transcript_search_next_prev_wrap() {
        let mut app = make_app();
        app.add_message(Role::User, "foo foo foo".to_string());
        app.transcript_search.query = "foo".to_string();
        app.recompute_transcript_search();
        app.transcript_search.current_match = Some(2);

        app.transcript_search_step(true);
        assert_eq!(app.transcript_search.current_match, Some(0));

        app.transcript_search_step(false);
        assert_eq!(app.transcript_search.current_match, Some(2));
    }

    #[test]
    fn test_modal_owner_priority_is_deterministic() {
        let mut app = make_app();
        app.model_picker.visible = true;
        app.onboarding_dialog.visible = true;

        assert_eq!(app.active_modal_owner(), Some(ModalOwner::Onboarding));
    }

    #[test]
    fn test_modal_active_prevents_chat_submit_leak() {
        let mut app = make_app();
        app.model_picker.visible = true;
        app.set_prompt_text("hello".to_string());

        let should_submit = app.handle_key_event(key_press(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!should_submit);
        assert_eq!(app.prompt_input.text, "hello");
    }
}
