// settings_screen.rs — Full-screen tabbed settings interface.
//
// Opened by /config or /settings commands. Provides a tabbed UI for
// viewing and editing General, Display, Privacy, Advanced, and KeyBindings
// settings. Changes are persisted via Settings::save_sync().

use crate::overlays::{
    centered_rect, render_dark_overlay, render_dialog_bg, MANGOCODE_ACCENT, MANGOCODE_MUTED,
    MANGOCODE_PANEL_BG, MANGOCODE_TEXT,
};
use mangocode_core::config::{Config, Settings};
use mangocode_core::keybindings::default_bindings;
use mangocode_core::output_styles::{all_styles_with_runtime_for_project, OutputStyleDef};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs, Wrap};
use ratatui::Frame;
use serde_json::Value;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsTab {
    General,
    Display,
    Privacy,
    Advanced,
    KeyBindings,
}

impl SettingsTab {
    pub fn all() -> &'static [SettingsTab] {
        &[
            SettingsTab::General,
            SettingsTab::Display,
            SettingsTab::Privacy,
            SettingsTab::Advanced,
            SettingsTab::KeyBindings,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsTab::General => "General",
            SettingsTab::Display => "Display",
            SettingsTab::Privacy => "Privacy",
            SettingsTab::Advanced => "Advanced",
            SettingsTab::KeyBindings => "KeyBindings",
        }
    }

    pub fn index(&self) -> usize {
        Self::all().iter().position(|t| t == self).unwrap_or(0)
    }
}

pub struct SettingsScreen {
    pub visible: bool,
    pub active_tab: SettingsTab,
    pub scroll_offset: u16,
    /// Which field is being edited (field name as key).
    pub edit_field: Option<String>,
    /// Current buffer content while editing a field.
    pub edit_value: String,
    /// Snapshot of settings at open time for display.
    pub settings_snapshot: Settings,
    /// Settings load failure captured at open time. When present, the screen
    /// remains read-only to avoid overwriting a malformed settings file.
    pub settings_load_error: Option<String>,
    /// One-shot status message for the owning app to surface.
    pub last_status_message: Option<String>,
    /// Pending changes (field_name → new_value string).
    pub pending_changes: std::collections::HashMap<String, String>,
    /// Path to the settings file backing this screen.
    settings_path: PathBuf,

    // ---- Real settings fields ----
    /// Whether auto-compact is enabled.
    pub auto_compact_enabled: bool,
    /// Auto-compact threshold (0-100%).
    pub auto_compact_threshold: u8,
    /// Whether desktop notifications are enabled.
    pub notifications_enabled: bool,
    /// Whether to reduce UI motion (animations).
    pub reduce_motion: bool,
    /// Whether to show turn duration in status bar.
    pub show_turn_duration: bool,
    /// Whether the terminal progress bar is enabled.
    pub terminal_progress_bar: bool,
    /// Whether telemetry is enabled (mirror of `!disableTelemetry`).
    pub telemetry_enabled: bool,
    /// Whether usage-data sharing is enabled (mirror of `shareUsageData`).
    pub usage_sharing_enabled: bool,

    /// Index of the currently-selected interactive field within the active tab.
    pub selected_field: usize,
}

// ---------------------------------------------------------------------------
// Interactive field model
//
// Every configurable item shown in a tab is described by a `Field`. The same
// list drives navigation (which rows the cursor stops on), activation (what
// Enter/Space/←/→ do) and rendering (which row is highlighted). Keeping a single
// source of truth guarantees that anything you can see, you can select and
// change.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FieldId {
    Model,
    MaxTokens,
    OutputStyle,
    AutoCompact,
    AutoCompactThreshold,
    Notifications,
    ShowTurnDuration,
    OutputFormat,
    Verbose,
    ReduceMotion,
    TerminalProgressBar,
    Telemetry,
    UsageSharing,
}

#[derive(Clone)]
enum FieldControl {
    /// Boolean toggle. Carries the current value.
    Bool(bool),
    /// Cycles through a fixed list of options. Carries the current value.
    Cycle { value: String, options: Vec<String> },
    /// Free-text / numeric value edited inline. Carries the current value.
    Text(String),
}

#[derive(Clone)]
struct Field {
    id: FieldId,
    label: &'static str,
    description: String,
    control: FieldControl,
}

/// The settings.json key used to edit a text field inline.
fn text_field_key(id: FieldId) -> &'static str {
    match id {
        FieldId::Model => "model",
        FieldId::MaxTokens => "max_tokens",
        FieldId::AutoCompactThreshold => "compact_threshold",
        _ => "",
    }
}

fn output_format_name(fmt: &mangocode_core::config::OutputFormat) -> String {
    use mangocode_core::config::OutputFormat;
    match fmt {
        OutputFormat::Text => "text".to_string(),
        OutputFormat::Json => "json".to_string(),
        OutputFormat::StreamJson => "stream-json".to_string(),
    }
}

fn output_format_options() -> Vec<String> {
    vec![
        "text".to_string(),
        "json".to_string(),
        "stream-json".to_string(),
    ]
}

fn output_format_from_name(name: &str) -> mangocode_core::config::OutputFormat {
    use mangocode_core::config::OutputFormat;
    match name {
        "json" => OutputFormat::Json,
        "stream-json" => OutputFormat::StreamJson,
        _ => OutputFormat::Text,
    }
}

/// Available output-style names, with "default" first.
fn output_style_options(config: &Config) -> Vec<String> {
    let mut opts = vec!["default".to_string()];
    for style in available_output_styles(config) {
        if !opts.iter().any(|o| o.eq_ignore_ascii_case(&style.name)) {
            opts.push(style.name);
        }
    }
    opts
}

fn apply_model_setting(config: &mut Config, value: &str) {
    if mangocode_api::normalize_model_id(value).is_none() {
        config.model = None;
        return;
    }

    mangocode_api::apply_model_selection_to_config(config, value, None);
}

fn apply_output_style_setting(config: &mut Config, value: &str) {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("default") {
        config.output_style = None;
        return;
    }

    let styles = available_output_styles(config);
    let canonical = mangocode_core::output_styles::find_style(&styles, value)
        .map(|style| style.name.clone())
        .unwrap_or_else(|| value.to_string());
    config.output_style = Some(canonical);
}

fn configured_output_style_name(config: &Config) -> &str {
    config
        .output_style
        .as_deref()
        .map(str::trim)
        .filter(|style| !style.is_empty())
        .unwrap_or("default")
}

fn available_output_styles(config: &Config) -> Vec<OutputStyleDef> {
    all_styles_with_runtime_for_project(&Settings::config_dir(), config.project_dir.as_deref())
}

fn sync_settings_snapshot_from_config(settings: &mut Settings, config: &Config) {
    settings.config = config.clone();
    settings.provider = config.provider.clone();
}

/// Effective auto-compact threshold as an integer percentage (1-100).
fn threshold_pct(config: &Config) -> u8 {
    (config.effective_compact_threshold() * 100.0)
        .round()
        .clamp(1.0, 100.0) as u8
}

impl SettingsScreen {
    pub fn new() -> Self {
        let settings_path = mangocode_core::config::Settings::global_settings_path();
        let (settings_snapshot, settings_load_error) =
            load_settings_snapshot_from_path(&settings_path);
        let screen = Self {
            visible: false,
            active_tab: SettingsTab::General,
            scroll_offset: 0,
            edit_field: None,
            edit_value: String::new(),
            settings_snapshot,
            settings_load_error,
            last_status_message: None,
            pending_changes: std::collections::HashMap::new(),
            settings_path,
            auto_compact_enabled: false,
            auto_compact_threshold: 90,
            notifications_enabled: true,
            reduce_motion: false,
            show_turn_duration: false,
            terminal_progress_bar: true,
            telemetry_enabled: false,
            usage_sharing_enabled: false,
            selected_field: 0,
        };
        screen
    }

    /// Refresh all mirrored UI-setting fields from the current snapshot/disk.
    /// Auto-compact + threshold are config-backed; the rest are top-level keys.
    /// Call once at startup (the live app reads these mirrors directly) and
    /// whenever the screen is opened.
    pub fn reload_ui_mirrors(&mut self) {
        self.auto_compact_enabled = self.settings_snapshot.config.auto_compact;
        self.auto_compact_threshold = threshold_pct(&self.settings_snapshot.config);
        self.notifications_enabled = read_setting_bool(&self.settings_path, "notifications", true);
        self.reduce_motion = read_setting_bool(&self.settings_path, "reduceMotion", false);
        self.show_turn_duration =
            read_setting_bool(&self.settings_path, "showTurnDuration", false);
        self.terminal_progress_bar =
            read_setting_bool(&self.settings_path, "terminalProgressBar", true);
        // Telemetry is stored inverted as `disableTelemetry` (default: disabled).
        self.telemetry_enabled = !read_setting_bool(&self.settings_path, "disableTelemetry", true);
        self.usage_sharing_enabled =
            read_setting_bool(&self.settings_path, "shareUsageData", false);
    }

    pub fn open(&mut self, live_config: &Config) {
        let (settings_snapshot, settings_load_error) =
            load_settings_snapshot_from_path(&self.settings_path);
        self.settings_snapshot = settings_snapshot;
        self.settings_load_error = settings_load_error;
        self.last_status_message = self
            .settings_load_error
            .as_deref()
            .map(settings_load_warning_message);
        self.settings_snapshot.config = live_config.clone();
        self.pending_changes.clear();
        self.edit_field = None;
        self.edit_value.clear();
        self.scroll_offset = 0;
        self.active_tab = SettingsTab::General;
        self.selected_field = 0;
        self.visible = true;

        // Wire real boolean settings from the (now live-config) snapshot.
        self.reload_ui_mirrors();
    }

    pub fn take_status_message(&mut self) -> Option<String> {
        self.last_status_message.take()
    }

    pub fn close(&mut self) {
        self.visible = false;
        self.edit_field = None;
        self.edit_value.clear();
    }

    pub fn next_tab(&mut self) {
        let idx = self.active_tab.index();
        let next = (idx + 1) % SettingsTab::all().len();
        self.active_tab = SettingsTab::all()[next].clone();
        self.scroll_offset = 0;
        self.selected_field = 0;
        self.edit_field = None;
        self.edit_value.clear();
    }

    pub fn prev_tab(&mut self) {
        let idx = self.active_tab.index();
        let prev = if idx == 0 {
            SettingsTab::all().len() - 1
        } else {
            idx - 1
        };
        self.active_tab = SettingsTab::all()[prev].clone();
        self.scroll_offset = 0;
        self.selected_field = 0;
        self.edit_field = None;
        self.edit_value.clear();
    }

    /// Move the cursor up. On tabs with interactive fields this moves the
    /// selection; on informational tabs (Advanced, KeyBindings) it scrolls the
    /// content.
    pub fn select_prev(&mut self) {
        if self.field_count() == 0 {
            self.scroll_offset = self.scroll_offset.saturating_sub(1);
        } else if self.selected_field > 0 {
            self.selected_field -= 1;
        }
    }

    /// Move the cursor down (see [`select_prev`]).
    pub fn select_next(&mut self) {
        let n = self.field_count();
        if n == 0 {
            self.scroll_offset = self.scroll_offset.saturating_add(1);
        } else if self.selected_field + 1 < n {
            self.selected_field += 1;
        }
    }

    pub fn page_up(&mut self) {
        if self.field_count() == 0 {
            self.scroll_offset = self.scroll_offset.saturating_sub(10);
        } else {
            self.selected_field = self.selected_field.saturating_sub(5);
        }
    }

    pub fn page_down(&mut self) {
        let n = self.field_count();
        if n == 0 {
            self.scroll_offset = self.scroll_offset.saturating_add(10);
        } else {
            self.selected_field = (self.selected_field + 5).min(n - 1);
        }
    }

    /// Number of interactive fields in the active tab.
    pub fn field_count(&self) -> usize {
        self.active_tab_fields().len()
    }

    /// Backwards-compatible alias retained for tests.
    #[cfg(test)]
    fn max_fields_for_tab(&self) -> usize {
        self.field_count()
    }

    /// Build the list of interactive fields for the active tab. This is the
    /// single source of truth shared by navigation, activation and rendering.
    fn active_tab_fields(&self) -> Vec<Field> {
        match &self.active_tab {
            SettingsTab::General => self.general_fields(),
            SettingsTab::Display => self.display_fields(),
            SettingsTab::Privacy => self.privacy_fields(),
            SettingsTab::Advanced | SettingsTab::KeyBindings => Vec::new(),
        }
    }

    fn general_fields(&self) -> Vec<Field> {
        let cfg = &self.settings_snapshot.config;
        let model = cfg
            .model
            .clone()
            .unwrap_or_else(|| mangocode_core::constants::DEFAULT_MODEL.to_string());
        let max_tokens = cfg
            .max_tokens
            .map(|n| n.to_string())
            .unwrap_or_else(|| mangocode_core::constants::DEFAULT_MAX_TOKENS.to_string());
        vec![
            Field {
                id: FieldId::Model,
                label: "Model",
                description: "AI model used for responses.".to_string(),
                control: FieldControl::Text(model),
            },
            Field {
                id: FieldId::MaxTokens,
                label: "Max Tokens",
                description: "Maximum tokens per response.".to_string(),
                control: FieldControl::Text(max_tokens),
            },
            Field {
                id: FieldId::OutputStyle,
                label: "Output Style",
                description: "Controls the verbosity and format of responses.".to_string(),
                control: FieldControl::Cycle {
                    value: configured_output_style_name(cfg).to_string(),
                    options: output_style_options(cfg),
                },
            },
            Field {
                id: FieldId::AutoCompact,
                label: "Auto-compact",
                description: "Summarize old turns automatically near the context limit."
                    .to_string(),
                control: FieldControl::Bool(self.auto_compact_enabled),
            },
            Field {
                id: FieldId::AutoCompactThreshold,
                label: "Auto-compact threshold",
                description: "Context-window % at which auto-compact triggers (1-100)."
                    .to_string(),
                control: FieldControl::Text(format!("{}%", self.auto_compact_threshold)),
            },
            Field {
                id: FieldId::Notifications,
                label: "Desktop notifications",
                description: "Notify when turn completes".to_string(),
                control: FieldControl::Bool(self.notifications_enabled),
            },
            Field {
                id: FieldId::ShowTurnDuration,
                label: "Show turn duration",
                description: "Display elapsed time per turn".to_string(),
                control: FieldControl::Bool(self.show_turn_duration),
            },
        ]
    }

    fn display_fields(&self) -> Vec<Field> {
        let cfg = &self.settings_snapshot.config;
        vec![
            Field {
                id: FieldId::OutputFormat,
                label: "Output Format",
                description: "Format for non-interactive output.".to_string(),
                control: FieldControl::Cycle {
                    value: output_format_name(&cfg.output_format),
                    options: output_format_options(),
                },
            },
            Field {
                id: FieldId::Verbose,
                label: "Verbose Mode",
                description: "Show additional debug information during queries.".to_string(),
                control: FieldControl::Bool(cfg.verbose),
            },
            Field {
                id: FieldId::ReduceMotion,
                label: "Reduce motion",
                description: "Disable UI animations".to_string(),
                control: FieldControl::Bool(self.reduce_motion),
            },
            Field {
                id: FieldId::TerminalProgressBar,
                label: "Terminal progress bar",
                description: "Show progress during tool use".to_string(),
                control: FieldControl::Bool(self.terminal_progress_bar),
            },
        ]
    }

    fn privacy_fields(&self) -> Vec<Field> {
        vec![
            Field {
                id: FieldId::Telemetry,
                label: "Telemetry",
                description: "Send anonymised usage statistics to help improve MangoCode."
                    .to_string(),
                control: FieldControl::Bool(self.telemetry_enabled),
            },
            Field {
                id: FieldId::UsageSharing,
                label: "Usage Sharing",
                description: "Share aggregate usage data for product improvement.".to_string(),
                control: FieldControl::Bool(self.usage_sharing_enabled),
            },
        ]
    }

    /// Persist a config-level change (theme, output format, verbose, output
    /// style) to the settings file, preserving unknown keys.
    fn persist_config(&mut self, config: &Config) -> Result<(), String> {
        if let Some(load_error) = self.settings_load_error.as_deref() {
            return Err(settings_save_blocked_message(load_error));
        }
        let mut next_settings = self.settings_snapshot.clone();
        sync_settings_snapshot_from_config(&mut next_settings, config);
        save_settings_snapshot_preserving_existing_keys(&self.settings_path, &next_settings)?;
        self.settings_snapshot = next_settings;
        Ok(())
    }

    /// Start editing a field by name, seeding the buffer with current value.
    pub fn start_edit(&mut self, field: &str, current_value: &str) {
        self.edit_field = Some(field.to_string());
        self.edit_value = current_value.to_string();
    }

    /// Commit the current edit to pending_changes.
    pub fn commit_edit(&mut self) {
        if let Some(field) = self.edit_field.take() {
            let value = std::mem::take(&mut self.edit_value);
            self.pending_changes.insert(field, value);
        }
    }

    /// Discard the current edit.
    pub fn cancel_edit(&mut self) {
        self.edit_field = None;
        self.edit_value.clear();
    }

    /// Apply all pending changes to settings and persist them.
    pub fn apply_and_save(&mut self, config: &mut Config) -> Result<(), String> {
        if let Some(load_error) = self.settings_load_error.as_deref() {
            return Err(settings_save_blocked_message(load_error));
        }

        let mut next_config = config.clone();
        for (field, value) in &self.pending_changes {
            match field.as_str() {
                "model" => {
                    apply_model_setting(&mut next_config, value);
                }
                "max_tokens" => {
                    if let Ok(n) = value.parse::<u32>() {
                        next_config.max_tokens = Some(n);
                    }
                }
                "output_style" => {
                    apply_output_style_setting(&mut next_config, value);
                }
                "compact_threshold" => {
                    // Accept "85", "85%", or "0.85"; store as a 0.0-1.0 fraction.
                    let raw = value.trim().trim_end_matches('%').trim();
                    if let Ok(n) = raw.parse::<f32>() {
                        let fraction = if n > 1.0 { n / 100.0 } else { n };
                        next_config.compact_threshold = fraction.clamp(0.01, 1.0);
                    }
                }
                _ => {}
            }
        }

        let mut next_settings = self.settings_snapshot.clone();
        sync_settings_snapshot_from_config(&mut next_settings, &next_config);
        save_settings_snapshot_preserving_existing_keys(&self.settings_path, &next_settings)?;

        *config = next_config;
        self.settings_snapshot = next_settings;
        self.auto_compact_threshold = threshold_pct(&self.settings_snapshot.config);
        self.pending_changes.clear();
        self.last_status_message = None;
        Ok(())
    }
}

impl Default for SettingsScreen {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Settings I/O helpers
// ---------------------------------------------------------------------------

fn load_settings_snapshot_from_path(path: &Path) -> (Settings, Option<String>) {
    if path == mangocode_core::config::Settings::global_settings_path() {
        return match Settings::load_sync() {
            Ok(settings) => (settings, None),
            Err(err) => (Settings::default(), Some(err.to_string())),
        };
    }

    if !path.exists() {
        return (Settings::default(), None);
    }

    match std::fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<Settings>(&content) {
            Ok(settings) => (settings, None),
            Err(err) => (
                Settings::default(),
                Some(format!("failed to parse {}: {}", path.display(), err)),
            ),
        },
        Err(err) => (
            Settings::default(),
            Some(format!("failed to read {}: {}", path.display(), err)),
        ),
    }
}

fn settings_load_warning_message(error: &str) -> String {
    format!("Settings file could not be loaded: {error}")
}

fn settings_save_blocked_message(error: &str) -> String {
    format!("Settings not saved because the existing settings file could not be loaded: {error}")
}

fn read_settings_json_value(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(Value::Object(Default::default()));
    }

    let content = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    serde_json::from_str(&content)
        .map_err(|err| format!("failed to parse {}: {}", path.display(), err))
}

fn merge_json_preserving_unknowns(existing: &mut Value, updates: Value) {
    match (existing, updates) {
        (Value::Object(existing_obj), Value::Object(update_obj)) => {
            for (key, update_value) in update_obj {
                if let Some(existing_value) = existing_obj.get_mut(&key) {
                    merge_json_preserving_unknowns(existing_value, update_value);
                } else {
                    existing_obj.insert(key, update_value);
                }
            }
        }
        (existing_value, update_value) => {
            *existing_value = update_value;
        }
    }
}

fn write_settings_json_value(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
    }
    let content = serde_json::to_string_pretty(value)
        .map_err(|err| format!("failed to serialize settings: {}", err))?;
    std::fs::write(path, content)
        .map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

fn save_settings_snapshot_preserving_existing_keys(
    path: &Path,
    settings: &Settings,
) -> Result<(), String> {
    let mut value = read_settings_json_value(path)?;
    let updates = serde_json::to_value(settings)
        .map_err(|err| format!("failed to serialize settings: {}", err))?;
    merge_json_preserving_unknowns(&mut value, updates);
    write_settings_json_value(path, &value)
}

/// Read a boolean value from `settings.json` by camelCase key, falling back to
/// `default` when the file is absent or the key is missing.
fn read_setting_bool(path: &Path, key: &str, default: bool) -> bool {
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(b) = val.get(key).and_then(|v| v.as_bool()) {
                return b;
            }
        }
    }
    default
}

/// Write a single boolean key-value pair to `settings.json`, preserving other
/// fields already present in the file.
fn save_setting_bool(path: &Path, key: &str, value: bool) -> Result<(), String> {
    let mut val = read_settings_json_value(path)?;
    if let Some(obj) = val.as_object_mut() {
        obj.insert(key.to_string(), serde_json::Value::Bool(value));
    } else {
        return Err(format!("{} must contain a JSON object", path.display()));
    }
    write_settings_json_value(path, &val)
}

/// Activate the currently-selected field. `dir` is the cycle direction
/// (`-1` previous, `+1` next); booleans toggle regardless of direction.
/// When `allow_edit` is true, activating a text field begins inline editing
/// (Enter); directional keys (←/→) pass `false` so they never open the editor.
pub fn activate_current_field(
    screen: &mut SettingsScreen,
    config: &mut Config,
    dir: i32,
    allow_edit: bool,
) {
    let fields = screen.active_tab_fields();
    let Some(field) = fields.get(screen.selected_field).cloned() else {
        return;
    };
    match field.control {
        FieldControl::Bool(_) => toggle_bool_field(screen, config, field.id),
        FieldControl::Cycle { value, options } => {
            cycle_field(screen, config, field.id, &value, &options, dir)
        }
        FieldControl::Text(current) => {
            if allow_edit {
                screen.start_edit(text_field_key(field.id), &current);
            }
        }
    }
}

/// Toggle the selected boolean field. Retained as a public entry point and
/// used by tests; no-op (with a blocked message) when settings failed to load.
pub fn toggle_current_field(screen: &mut SettingsScreen, config: &mut Config) {
    if let Some(load_error) = screen.settings_load_error.as_deref() {
        screen.last_status_message = Some(settings_save_blocked_message(load_error));
        return;
    }
    activate_current_field(screen, config, 1, true);
}

/// Toggle a boolean field and persist it. Top-level keys are written directly;
/// `Verbose` and `Auto-compact` live in the config object (so the query engine
/// honors them per turn).
fn toggle_bool_field(screen: &mut SettingsScreen, config: &mut Config, id: FieldId) {
    if let Some(load_error) = screen.settings_load_error.as_deref() {
        screen.last_status_message = Some(settings_save_blocked_message(load_error));
        return;
    }

    // Verbose lives in the config object, persisted via the snapshot.
    if id == FieldId::Verbose {
        let next = !config.verbose;
        config.verbose = next;
        match screen.persist_config(config) {
            Ok(()) => screen.last_status_message = None,
            Err(err) => {
                config.verbose = !next; // roll back live value on failure
                screen.last_status_message = Some(format!("Settings not saved: {err}"));
            }
        }
        return;
    }

    // Auto-compact lives in config.auto_compact so the query engine gates on it.
    if id == FieldId::AutoCompact {
        let next = !screen.auto_compact_enabled;
        let prev = config.auto_compact;
        config.auto_compact = next;
        match screen.persist_config(config) {
            Ok(()) => {
                screen.auto_compact_enabled = next;
                screen.last_status_message = None;
            }
            Err(err) => {
                config.auto_compact = prev; // roll back live value on failure
                screen.last_status_message = Some(format!("Settings not saved: {err}"));
            }
        }
        return;
    }

    // Everything else is a top-level settings.json key.
    // (key, current value, stored-inverted)
    let (key, current, inverted) = match id {
        FieldId::Notifications => ("notifications", screen.notifications_enabled, false),
        FieldId::ShowTurnDuration => ("showTurnDuration", screen.show_turn_duration, false),
        FieldId::ReduceMotion => ("reduceMotion", screen.reduce_motion, false),
        FieldId::TerminalProgressBar => ("terminalProgressBar", screen.terminal_progress_bar, false),
        FieldId::UsageSharing => ("shareUsageData", screen.usage_sharing_enabled, false),
        FieldId::Telemetry => ("disableTelemetry", screen.telemetry_enabled, true),
        _ => return,
    };

    let next = !current;
    let stored = if inverted { !next } else { next };
    match save_setting_bool(&screen.settings_path, key, stored) {
        Ok(()) => {
            match id {
                FieldId::AutoCompact => screen.auto_compact_enabled = next,
                FieldId::Notifications => screen.notifications_enabled = next,
                FieldId::ShowTurnDuration => screen.show_turn_duration = next,
                FieldId::ReduceMotion => screen.reduce_motion = next,
                FieldId::TerminalProgressBar => screen.terminal_progress_bar = next,
                FieldId::UsageSharing => screen.usage_sharing_enabled = next,
                FieldId::Telemetry => screen.telemetry_enabled = next,
                _ => {}
            }
            screen.last_status_message = None;
        }
        Err(err) => screen.last_status_message = Some(format!("Settings not saved: {err}")),
    }
}

/// Cycle an enum field to the next/previous option and persist it.
fn cycle_field(
    screen: &mut SettingsScreen,
    config: &mut Config,
    id: FieldId,
    value: &str,
    options: &[String],
    dir: i32,
) {
    if let Some(load_error) = screen.settings_load_error.as_deref() {
        screen.last_status_message = Some(settings_save_blocked_message(load_error));
        return;
    }
    if options.is_empty() {
        return;
    }

    let len = options.len();
    let idx = options
        .iter()
        .position(|o| o.eq_ignore_ascii_case(value))
        .unwrap_or(0);
    let next_idx = if dir < 0 {
        (idx + len - 1) % len
    } else {
        (idx + 1) % len
    };
    let choice = options[next_idx].clone();

    let previous = config.clone();
    match id {
        FieldId::OutputFormat => config.output_format = output_format_from_name(&choice),
        FieldId::OutputStyle => apply_output_style_setting(config, &choice),
        _ => return,
    }

    match screen.persist_config(config) {
        Ok(()) => {
            screen.last_status_message = Some(format!("{} set to {}", field_label(id), choice));
        }
        Err(err) => {
            *config = previous; // roll back live value on failure
            screen.last_status_message = Some(format!("Settings not saved: {err}"));
        }
    }
}

fn field_label(id: FieldId) -> &'static str {
    match id {
        FieldId::OutputFormat => "Output Format",
        FieldId::OutputStyle => "Output Style",
        FieldId::Verbose => "Verbose Mode",
        _ => "Setting",
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the settings screen (full-screen popup) into `frame`.
pub fn render_settings_screen(frame: &mut Frame, screen: &SettingsScreen, area: Rect) {
    if !screen.visible {
        return;
    }

    render_dark_overlay(frame, area);

    // 80% width, 90% height, centred
    let w = (area.width * 4 / 5)
        .max(60)
        .min(area.width.saturating_sub(2));
    let h = (area.height * 9 / 10)
        .max(20)
        .min(area.height.saturating_sub(2));
    let popup = centered_rect(w, h, area);
    render_dialog_bg(frame, popup);

    // Inset inner area
    let inner = Rect {
        x: popup.x + 2,
        y: popup.y + 1,
        width: popup.width.saturating_sub(4),
        height: popup.height.saturating_sub(2),
    };

    if inner.height < 6 {
        return;
    }

    // Split into header + tabs + content + footer
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let header_area = layout[0];
    let tabs_area = layout[1];
    let content_area = layout[2];
    let footer_area = layout[3];

    let title = Line::from(vec![
        Span::styled(
            " Settings",
            Style::default()
                .fg(MANGOCODE_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" — MangoCode", Style::default().fg(MANGOCODE_MUTED)),
        Span::styled(
            format!(
                "{:>width$}",
                "Esc close",
                width = inner.width.saturating_sub(19) as usize
            ),
            Style::default().fg(MANGOCODE_MUTED),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title).style(Style::default().bg(MANGOCODE_PANEL_BG)),
        header_area,
    );

    // Tabs bar
    let tab_labels: Vec<Line> = SettingsTab::all()
        .iter()
        .map(|t| {
            if *t == screen.active_tab {
                Line::from(vec![Span::styled(
                    format!(" {} ", t.label()),
                    Style::default()
                        .fg(Color::Black)
                        .bg(MANGOCODE_ACCENT)
                        .add_modifier(Modifier::BOLD),
                )])
            } else {
                Line::from(vec![Span::styled(
                    format!(" {} ", t.label()),
                    Style::default().fg(MANGOCODE_MUTED),
                )])
            }
        })
        .collect();

    let tabs = Tabs::new(tab_labels)
        .divider(Span::styled("  ", Style::default().fg(MANGOCODE_MUTED)))
        .style(Style::default().fg(MANGOCODE_MUTED).bg(MANGOCODE_PANEL_BG));
    frame.render_widget(tabs, tabs_area);

    // Tab content
    render_tab_content(frame, screen, content_area);

    // Footer
    let footer = if screen.edit_field.is_some() {
        Line::from(vec![
            Span::styled(
                " Enter ",
                Style::default()
                    .fg(MANGOCODE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("save  "),
            Span::styled(
                " Esc ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("cancel"),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                " Tab ",
                Style::default()
                    .fg(MANGOCODE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("tab  "),
            Span::styled(
                " ↑↓ ",
                Style::default()
                    .fg(MANGOCODE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("select  "),
            Span::styled(
                " ←→/Space ",
                Style::default()
                    .fg(MANGOCODE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("change  "),
            Span::styled(
                " Enter ",
                Style::default()
                    .fg(MANGOCODE_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("toggle/edit  "),
            Span::styled(
                " Esc ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("close"),
        ])
    };
    let footer_para = Paragraph::new(vec![footer])
        .style(Style::default().fg(MANGOCODE_MUTED).bg(MANGOCODE_PANEL_BG))
        .alignment(Alignment::Center);
    frame.render_widget(footer_para, footer_area);
}

fn render_tab_content(frame: &mut Frame, screen: &SettingsScreen, area: Rect) {
    let mut selected_line: Option<u16> = None;
    let mut lines = screen.build_tab_lines(&mut selected_line);

    if let Some(load_error) = screen.settings_load_error.as_deref() {
        let mut warning = vec![
            Line::from(vec![Span::styled(
                "Settings file could not be loaded. Changes are disabled until it is fixed.",
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                load_error.to_string(),
                Style::default().fg(Color::LightRed),
            )]),
            Line::from(""),
        ];
        let warning_len = warning.len() as u16;
        warning.extend(lines);
        lines = warning;
        selected_line = selected_line.map(|l| l + warning_len);
    }

    let total = lines.len() as u16;
    let visible = area.height.max(1);
    let max_scroll = total.saturating_sub(visible);

    // On interactive tabs the view follows the selected field; on informational
    // tabs the manual scroll offset is used.
    let scroll = if screen.field_count() == 0 {
        screen.scroll_offset.min(max_scroll)
    } else if let Some(sl) = selected_line {
        let anchor = if visible > 6 { sl.saturating_sub(3) } else { sl };
        anchor.min(max_scroll)
    } else {
        0
    };

    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(para, area);
}

// ---------------------------------------------------------------------------
// General tab
// ---------------------------------------------------------------------------

impl SettingsScreen {
    /// Build the display lines for the active tab. Records the line index of the
    /// currently-selected field in `sel_line` so the view can scroll to follow it.
    fn build_tab_lines(&self, sel_line: &mut Option<u16>) -> Vec<Line<'static>> {
        match &self.active_tab {
            SettingsTab::General => self.build_general(sel_line),
            SettingsTab::Display => self.build_display(sel_line),
            SettingsTab::Privacy => self.build_privacy(sel_line),
            SettingsTab::Advanced => build_advanced_lines(self),
            SettingsTab::KeyBindings => build_keybindings_lines(self),
        }
    }

    /// Render one interactive field's rows, highlighting it when selected and
    /// recording its line index into `sel_line`.
    fn push_field(
        &self,
        lines: &mut Vec<Line<'static>>,
        fields: &[Field],
        fi: usize,
        sel_line: &mut Option<u16>,
    ) {
        let Some(field) = fields.get(fi) else {
            return;
        };
        let selected = self.selected_field == fi;
        if selected {
            *sel_line = Some(lines.len() as u16);
        }
        let editing = match &field.control {
            FieldControl::Text(_)
                if self.edit_field.as_deref() == Some(text_field_key(field.id)) =>
            {
                Some(self.edit_value.as_str())
            }
            _ => None,
        };
        lines.extend(render_field(field, selected, editing));
    }

    fn build_general(&self, sel_line: &mut Option<u16>) -> Vec<Line<'static>> {
        let cfg = &self.settings_snapshot.config;
        let fields = self.general_fields();
        let mut lines: Vec<Line<'static>> = Vec::new();

        lines.push(section_header("General Settings"));
        lines.push(Line::from(""));

        // Model (+ available models hint)
        self.push_field(&mut lines, &fields, 0, sel_line);
        lines.push(indent_line(
            "      Available: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001",
            Color::DarkGray,
        ));
        lines.push(Line::from(""));

        // Max tokens
        self.push_field(&mut lines, &fields, 1, sel_line);
        lines.push(Line::from(""));

        // Output style (+ available styles hint)
        self.push_field(&mut lines, &fields, 2, sel_line);
        let style_names: Vec<String> = available_output_styles(cfg)
            .into_iter()
            .map(|s| s.name)
            .collect();
        lines.push(indent_line(
            &format!("      Available: {}", style_names.join(", ")),
            Color::DarkGray,
        ));
        lines.push(Line::from(""));

        // Working directory (informational)
        let wd = cfg
            .project_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "(unknown)".to_string())
            });
        lines.push(label_value_line("Working Directory", &wd));
        lines.push(indent_line(
            "      (Set via --project-dir flag)",
            Color::DarkGray,
        ));
        lines.push(Line::from(""));

        lines.push(section_header("Behaviour"));
        lines.push(Line::from(""));
        self.push_field(&mut lines, &fields, 3, sel_line); // Auto-compact
        self.push_field(&mut lines, &fields, 4, sel_line); // Auto-compact threshold
        self.push_field(&mut lines, &fields, 5, sel_line); // Desktop notifications
        self.push_field(&mut lines, &fields, 6, sel_line); // Show turn duration

        lines
    }

    fn build_display(&self, sel_line: &mut Option<u16>) -> Vec<Line<'static>> {
        let cfg = &self.settings_snapshot.config;
        let fields = self.display_fields();
        let mut lines: Vec<Line<'static>> = Vec::new();

        lines.push(section_header("Display Settings"));
        lines.push(Line::from(""));

        // Output format
        self.push_field(&mut lines, &fields, 0, sel_line);
        lines.push(indent_line(
            "      Options: text, json, stream-json",
            Color::DarkGray,
        ));
        lines.push(Line::from(""));

        // Verbose
        self.push_field(&mut lines, &fields, 1, sel_line);
        lines.push(Line::from(""));

        lines.push(section_header("UI Options"));
        lines.push(Line::from(""));
        self.push_field(&mut lines, &fields, 2, sel_line); // Reduce motion
        self.push_field(&mut lines, &fields, 3, sel_line); // Terminal progress bar
        lines.push(Line::from(""));

        // Output styles reference list
        lines.push(section_header("Available Output Styles"));
        lines.push(Line::from(""));
        let current_style = configured_output_style_name(cfg);
        for style in available_output_styles(cfg) {
            let active = current_style.eq_ignore_ascii_case(&style.name);
            let marker = if active { " *" } else { "  " };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}  {:<15}", marker, style.name),
                    Style::default()
                        .fg(if active {
                            MANGOCODE_ACCENT
                        } else {
                            MANGOCODE_TEXT
                        })
                        .add_modifier(if active {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(style.description.clone(), Style::default().fg(Color::DarkGray)),
            ]));
        }
        lines.push(Line::from(""));
        lines
    }

    fn build_privacy(&self, sel_line: &mut Option<u16>) -> Vec<Line<'static>> {
        let fields = self.privacy_fields();
        let mut lines: Vec<Line<'static>> = Vec::new();

        lines.push(section_header("Privacy Settings"));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  These settings control data sharing with Anthropic.",
            Style::default().fg(Color::DarkGray),
        )]));
        lines.push(Line::from(""));

        // Usage policy agreement (informational)
        let privacy = PrivacySnapshot::load();
        let agreed_label = match privacy.has_agreed {
            Some(true) => "Agreed",
            Some(false) => "Not agreed",
            None => "Unknown (field absent)",
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<25}", "Usage Policy"),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                agreed_label.to_string(),
                Style::default().fg(if privacy.has_agreed == Some(true) {
                    Color::Green
                } else {
                    Color::Yellow
                }),
            ),
        ]));
        lines.push(indent_line(
            "      Whether you have agreed to Anthropic's usage policy.",
            Color::DarkGray,
        ));
        lines.push(Line::from(""));

        lines.push(section_header("Data Sharing"));
        lines.push(Line::from(""));
        self.push_field(&mut lines, &fields, 0, sel_line); // Telemetry
        self.push_field(&mut lines, &fields, 1, sel_line); // Usage Sharing
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  Changes here save immediately and mirror /privacy.",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::ITALIC),
        )]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "  For full privacy policy see: https://www.anthropic.com/privacy",
            Style::default().fg(Color::DarkGray),
        )]));

        lines
    }
}

/// Render one field as a highlighted/selectable row plus a description line.
fn render_field(field: &Field, selected: bool, editing: Option<&str>) -> Vec<Line<'static>> {
    let label = field.label;
    let sel_style = Style::default()
        .fg(Color::Black)
        .bg(MANGOCODE_ACCENT)
        .add_modifier(Modifier::BOLD);

    let row = match &field.control {
        FieldControl::Bool(value) => {
            let (marker, state) = if *value { ("[x]", "ON") } else { ("[ ]", "OFF") };
            let text = format!("  {} {:<26} {}", marker, label, state);
            let style = if selected {
                sel_style
            } else {
                Style::default()
                    .fg(if *value { Color::Green } else { Color::DarkGray })
                    .add_modifier(if *value {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    })
            };
            Line::from(Span::styled(text, style))
        }
        FieldControl::Cycle { value, .. } => {
            let text = format!("  {:<26} \u{2039} {} \u{203a}", format!("{}:", label), value);
            let style = if selected {
                sel_style
            } else {
                Style::default().fg(MANGOCODE_TEXT)
            };
            Line::from(Span::styled(text, style))
        }
        FieldControl::Text(value) => {
            let (shown, hint) = match editing {
                Some(buf) => (format!("{}_", buf), "  [editing]"),
                None if selected => (value.clone(), "  (Enter to edit)"),
                None => (value.clone(), ""),
            };
            let text = format!("  {:<26} {}{}", format!("{}:", label), shown, hint);
            let style = if editing.is_some() {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else if selected {
                sel_style
            } else {
                Style::default().fg(MANGOCODE_TEXT)
            };
            Line::from(Span::styled(text, style))
        }
    };

    let desc = Line::from(vec![Span::styled(
        format!("      {}", field.description),
        Style::default().fg(Color::DarkGray),
    )]);

    vec![row, desc]
}

// Free-function wrappers retained for unit tests.
#[cfg(test)]
fn build_general_lines(screen: &SettingsScreen) -> Vec<Line<'static>> {
    screen.build_general(&mut None)
}

#[cfg(test)]
fn build_display_lines(screen: &SettingsScreen) -> Vec<Line<'static>> {
    screen.build_display(&mut None)
}

// ---------------------------------------------------------------------------
// Privacy tab
// ---------------------------------------------------------------------------

/// Privacy-relevant fields parsed from the raw settings JSON.
#[derive(Debug, Clone, Default)]
struct PrivacySnapshot {
    /// `hasAgreedToUsagePolicy` from settings.json
    has_agreed: Option<bool>,
}

impl PrivacySnapshot {
    /// Load privacy fields from `~/.mangocode/settings.json`.
    fn load() -> Self {
        let path = mangocode_core::config::Settings::global_settings_path();
        let Ok(content) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
            return Self::default();
        };
        Self {
            has_agreed: json.get("hasAgreedToUsagePolicy").and_then(|v| v.as_bool()),
        }
    }
}

// ---------------------------------------------------------------------------
// Advanced tab
// ---------------------------------------------------------------------------

fn build_advanced_lines(screen: &SettingsScreen) -> Vec<Line<'static>> {
    let cfg = &screen.settings_snapshot.config;
    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.push(section_header("Advanced Settings"));
    lines.push(Line::from(""));
    lines.push(indent_line(
        "  Read-only overview. Manage these via provider setup, /mcp, /hooks,",
        Color::DarkGray,
    ));
    lines.push(indent_line(
        "  and environment variables. Use ↑↓/PgUp/PgDn to scroll.",
        Color::DarkGray,
    ));
    lines.push(Line::from(""));

    // API key source
    let key_source = if cfg.api_key.is_some() {
        "config file (masked)"
    } else if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        "environment variable (ANTHROPIC_API_KEY)"
    } else {
        "not set"
    };
    lines.push(label_value_line("API Key Source", key_source));
    if cfg.api_key.is_some() {
        lines.push(indent_line("  sk-ant-api03-***...***", Color::DarkGray));
    }
    lines.push(Line::from(""));

    // MCP Servers
    lines.push(section_header("MCP Servers"));
    lines.push(Line::from(""));
    if cfg.mcp_servers.is_empty() {
        lines.push(indent_line("  (none configured)", Color::DarkGray));
    } else {
        for srv in &cfg.mcp_servers {
            let kind = if srv.server_type == "pipedream" {
                "pipedream"
            } else if srv.url.is_some() {
                "http"
            } else {
                &srv.server_type
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<20}", srv.name),
                    Style::default()
                        .fg(MANGOCODE_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("[{}]", kind), Style::default().fg(Color::DarkGray)),
            ]));
            if let Some(cmd) = &srv.command {
                lines.push(indent_line(&format!("  cmd: {}", cmd), Color::DarkGray));
            }
            if let Some(url) = &srv.url {
                lines.push(indent_line(&format!("  url: {}", url), Color::DarkGray));
            }
        }
    }
    lines.push(Line::from(""));

    // Hooks
    lines.push(section_header("Configured Hooks"));
    lines.push(Line::from(""));
    if cfg.hooks.is_empty() {
        lines.push(indent_line("  (none configured)", Color::DarkGray));
    } else {
        for (event, entries) in &cfg.hooks {
            for entry in entries {
                let event_name = format!("{:?}", event);
                let filter = entry
                    .tool_filter
                    .as_deref()
                    .map(|f| format!("[{}]", f))
                    .unwrap_or_default();
                let blocking = if entry.blocking { " (blocking)" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {:<20}", event_name),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(filter, Style::default().fg(MANGOCODE_ACCENT)),
                    Span::styled(blocking.to_string(), Style::default().fg(Color::Red)),
                ]));
                lines.push(indent_line(
                    &format!("    cmd: {}", entry.command),
                    Color::DarkGray,
                ));
            }
        }
    }
    lines.push(Line::from(""));

    // Environment variables
    lines.push(section_header("Environment Variables"));
    lines.push(Line::from(""));
    if cfg.env.is_empty() {
        lines.push(indent_line("  (none configured)", Color::DarkGray));
    } else {
        for key in cfg.env.keys() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<25}", key),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("= ***".to_string(), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }
    lines.push(Line::from(""));

    lines
}

// ---------------------------------------------------------------------------
// KeyBindings tab
// ---------------------------------------------------------------------------

fn build_keybindings_lines(_screen: &SettingsScreen) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = vec![
        section_header("Key Bindings"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Edit ~/.mangocode/keybindings.json to customise bindings.",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::ITALIC),
        )]),
        Line::from(""),
    ];

    // Group bindings by context
    let mut by_context: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();

    for binding in default_bindings() {
        if let Some(action) = &binding.action {
            let ctx_name = format!("{:?}", binding.context);
            let chord_str = binding
                .chord
                .iter()
                .map(|ks| {
                    let mut parts = Vec::new();
                    if ks.ctrl {
                        parts.push("Ctrl");
                    }
                    if ks.alt {
                        parts.push("Alt");
                    }
                    if ks.shift {
                        parts.push("Shift");
                    }
                    parts.push(ks.key.as_str());
                    parts.join("+")
                })
                .collect::<Vec<_>>()
                .join(" ");
            by_context
                .entry(ctx_name)
                .or_default()
                .push((chord_str, action.clone()));
        }
    }

    // Render in sorted context order
    let mut contexts: Vec<String> = by_context.keys().cloned().collect();
    contexts.sort();

    // Ensure Global and Chat come first
    contexts.retain(|c| c != "Global" && c != "Chat");
    let mut ordered = vec!["Global".to_string(), "Chat".to_string()];
    ordered.extend(contexts);

    for ctx in &ordered {
        if let Some(entries) = by_context.get(ctx) {
            lines.push(Line::from(vec![Span::styled(
                format!("  {} Context", ctx),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )]));
            lines.push(Line::from(""));
            for (chord, action) in entries {
                lines.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("{:<25}", chord),
                        Style::default()
                            .fg(MANGOCODE_ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(action.clone(), Style::default().fg(Color::White)),
                ]));
            }
            lines.push(Line::from(""));
        }
    }

    lines
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn section_header(title: &str) -> Line<'static> {
    Line::from(vec![Span::styled(
        format!("  {}", title),
        Style::default()
            .fg(MANGOCODE_ACCENT)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    )])
}

fn label_value_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {:<25}", label),
            Style::default()
                .fg(MANGOCODE_TEXT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(value.to_string(), Style::default().fg(MANGOCODE_ACCENT)),
    ])
}

fn indent_line(text: &str, color: Color) -> Line<'static> {
    Line::from(vec![Span::styled(
        text.to_string(),
        Style::default().fg(color),
    )])
}

// ---------------------------------------------------------------------------
// Key handling helpers (called from app.rs)
// ---------------------------------------------------------------------------

/// Returns `true` if the key event was consumed by the settings screen.
pub fn handle_settings_key(
    screen: &mut SettingsScreen,
    config: &mut Config,
    key: crossterm::event::KeyEvent,
) -> bool {
    use crossterm::event::{KeyCode, KeyModifiers};

    if !screen.visible {
        return false;
    }

    // Editing mode
    if screen.edit_field.is_some() {
        match key.code {
            KeyCode::Enter => {
                screen.commit_edit();
                if let Err(err) = screen.apply_and_save(config) {
                    screen.last_status_message = Some(err);
                }
            }
            KeyCode::Esc => {
                screen.cancel_edit();
            }
            KeyCode::Backspace => {
                screen.edit_value.pop();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                screen.edit_value.push(c);
            }
            _ => {}
        }
        return true;
    }

    // Navigation mode
    match key.code {
        KeyCode::Esc => {
            screen.close();
        }
        KeyCode::Tab => {
            screen.next_tab();
        }
        KeyCode::BackTab => {
            screen.prev_tab();
        }
        KeyCode::Up => {
            screen.select_prev();
        }
        KeyCode::Down => {
            screen.select_next();
        }
        KeyCode::PageUp => {
            screen.page_up();
        }
        KeyCode::PageDown => {
            screen.page_down();
        }
        // ←/→ change the selected field (cycle enums, toggle bools) but never
        // open the inline text editor.
        KeyCode::Left => {
            activate_current_field(screen, config, -1, false);
        }
        KeyCode::Right => {
            activate_current_field(screen, config, 1, false);
        }
        // Space changes the selected field but does not open the text editor.
        KeyCode::Char(' ') => {
            activate_current_field(screen, config, 1, false);
        }
        // Enter toggles/cycles, or begins editing a text field.
        KeyCode::Enter => {
            activate_current_field(screen, config, 1, true);
        }
        _ => {}
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    // Helper: create a temporary settings.json with a given JSON body.
    fn write_temp_settings(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(json.as_bytes()).expect("write");
        f
    }

    fn settings_screen_with_temp_settings(json: &str) -> (SettingsScreen, NamedTempFile) {
        let file = write_temp_settings(json);
        let mut screen = SettingsScreen::new();
        screen.settings_path = file.path().to_path_buf();
        let (settings_snapshot, settings_load_error) =
            load_settings_snapshot_from_path(&screen.settings_path);
        screen.settings_snapshot = settings_snapshot;
        screen.settings_load_error = settings_load_error;
        screen.last_status_message = None;
        screen.reload_ui_mirrors();
        (screen, file)
    }

    // ---------------------------------------------------------------------------
    // read_setting_bool tests
    // ---------------------------------------------------------------------------

    #[test]
    fn read_setting_bool_returns_default_when_file_missing() {
        let dir = TempDir::new().unwrap();
        let result = read_setting_bool(
            &dir.path().join("missing-settings.json"),
            "thisKeyDoesNotExist_xyzzy",
            true,
        );
        assert!(result, "default should be returned for missing key");
    }

    #[test]
    fn read_setting_bool_false_default_when_key_absent() {
        let file = write_temp_settings(r#"{"someOtherKey":true}"#);
        let result = read_setting_bool(file.path(), "anotherMissingKey_abc123", false);
        assert!(!result);
    }

    #[test]
    fn write_temp_settings_writes_json_body() {
        let file = write_temp_settings(r#"{"theme":"dark","model":"claude-3-opus"}"#);
        let content = std::fs::read_to_string(file.path()).expect("read temp settings");
        let value: serde_json::Value = serde_json::from_str(&content).expect("parse temp settings");
        assert_eq!(value["theme"], "dark");
        assert_eq!(value["model"], "claude-3-opus");
    }

    // ---------------------------------------------------------------------------
    // SettingsScreen::new() defaults
    // ---------------------------------------------------------------------------

    #[test]
    fn settings_screen_new_has_sensible_defaults() {
        let screen = SettingsScreen::new();
        assert!(!screen.visible, "should not be visible on creation");
        assert_eq!(screen.active_tab, SettingsTab::General);
        assert_eq!(screen.scroll_offset, 0);
        assert_eq!(screen.selected_field, 0);
        assert!(screen.edit_field.is_none());
        assert!(screen.edit_value.is_empty());
        assert!(screen.pending_changes.is_empty());
        // Boolean defaults
        assert!(screen.notifications_enabled, "notifications default on");
        assert!(!screen.reduce_motion, "reduce_motion default off");
        assert!(!screen.show_turn_duration, "show_turn_duration default off");
        assert!(
            screen.terminal_progress_bar,
            "terminal_progress_bar default on"
        );
    }

    #[test]
    fn settings_screen_new_auto_compact_threshold_sensible() {
        let screen = SettingsScreen::new();
        // Threshold must be in 0-100 range
        assert!(
            screen.auto_compact_threshold <= 100,
            "threshold {} out of range",
            screen.auto_compact_threshold
        );
    }

    #[test]
    fn settings_screen_open_uses_live_config_snapshot() {
        let project_dir = TempDir::new().unwrap();
        let mut screen = SettingsScreen::new();
        let config = Config {
            project_dir: Some(project_dir.path().to_path_buf()),
            output_style: Some("ProjectUI".to_string()),
            model: Some("openai/gpt-4o".to_string()),
            ..Default::default()
        };

        screen.open(&config);

        assert_eq!(
            screen.settings_snapshot.config.project_dir.as_deref(),
            Some(project_dir.path())
        );
        assert_eq!(
            screen.settings_snapshot.config.output_style.as_deref(),
            Some("ProjectUI")
        );
        assert_eq!(
            screen.settings_snapshot.config.model.as_deref(),
            Some("openai/gpt-4o")
        );
    }

    #[test]
    fn model_setting_trims_and_updates_provider() {
        let mut config = Config {
            provider: Some("anthropic".to_string()),
            ..Default::default()
        };

        apply_model_setting(&mut config, " openai/gpt-4o ");

        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn model_setting_clears_blank_model() {
        let mut config = Config {
            provider: Some("openai".to_string()),
            model: Some("gpt-4o".to_string()),
            ..Default::default()
        };

        apply_model_setting(&mut config, "   ");

        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model, None);
    }

    #[test]
    fn settings_snapshot_sync_mirrors_top_level_provider() {
        let mut settings = Settings {
            provider: Some("anthropic".to_string()),
            ..Default::default()
        };
        let config = Config {
            provider: Some("openai".to_string()),
            model: Some("openai/gpt-4o".to_string()),
            ..Default::default()
        };

        sync_settings_snapshot_from_config(&mut settings, &config);

        assert_eq!(settings.provider.as_deref(), Some("openai"));
        assert_eq!(settings.config.provider.as_deref(), Some("openai"));
        assert_eq!(settings.config.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn settings_json_merge_preserves_unknown_keys() {
        let mut existing = serde_json::json!({
            "autoCompact": true,
            "unknownTopLevel": "keep",
            "config": {
                "model": "old-model",
                "unknownNested": "keep"
            }
        });
        let settings = Settings {
            provider: Some("openai".to_string()),
            config: Config {
                model: Some("openai/gpt-4o".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let updates = serde_json::to_value(&settings).unwrap();

        merge_json_preserving_unknowns(&mut existing, updates);

        assert_eq!(existing["autoCompact"], true);
        assert_eq!(existing["unknownTopLevel"], "keep");
        assert_eq!(existing["config"]["unknownNested"], "keep");
        assert_eq!(existing["config"]["model"], "openai/gpt-4o");
        assert_eq!(existing["provider"], "openai");
    }

    #[test]
    fn apply_and_save_blocks_when_settings_load_failed() {
        let mut screen = SettingsScreen::new();
        screen.settings_load_error = Some("bad json".to_string());
        screen
            .pending_changes
            .insert("model".to_string(), "openai/gpt-4o".to_string());
        let mut config = Config {
            model: Some("old-model".to_string()),
            ..Default::default()
        };

        let err = screen.apply_and_save(&mut config).unwrap_err();

        assert!(err.contains("Settings not saved"));
        assert_eq!(config.model.as_deref(), Some("old-model"));
        assert_eq!(
            screen.pending_changes.get("model").map(String::as_str),
            Some("openai/gpt-4o")
        );
    }

    #[test]
    fn output_style_setting_trims_and_clears_default() {
        let mut config = Config::default();

        apply_output_style_setting(&mut config, " concise ");
        assert_eq!(config.output_style.as_deref(), Some("concise"));

        apply_output_style_setting(&mut config, " default ");
        assert_eq!(config.output_style, None);

        apply_output_style_setting(&mut config, "   ");
        assert_eq!(config.output_style, None);
    }

    #[test]
    fn output_style_setting_canonicalizes_project_style_case() {
        let project_dir = TempDir::new().unwrap();
        let styles_dir = project_dir.path().join(".mangocode").join("output-styles");
        std::fs::create_dir_all(&styles_dir).unwrap();
        std::fs::write(
            styles_dir.join("ProjectUI.md"),
            "# Project UI\nProject display style.\n\nUse project UI style.",
        )
        .unwrap();
        let mut config = Config {
            project_dir: Some(project_dir.path().to_path_buf()),
            ..Default::default()
        };

        apply_output_style_setting(&mut config, " projectui ");

        assert_eq!(config.output_style.as_deref(), Some("ProjectUI"));
    }

    // ---------------------------------------------------------------------------
    // toggle_current_field tests
    // ---------------------------------------------------------------------------

    #[test]
    fn toggle_current_field_general_field0_flips_auto_compact() {
        let (mut screen, _settings_file) =
            settings_screen_with_temp_settings(r#"{"autoCompact":true}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 3; // Auto-compact

        let before = screen.auto_compact_enabled;
        toggle_current_field(&mut screen, &mut config);
        assert_eq!(
            screen.auto_compact_enabled, !before,
            "auto_compact_enabled should have flipped"
        );
        // Toggle back
        toggle_current_field(&mut screen, &mut config);
        assert_eq!(screen.auto_compact_enabled, before);
    }

    #[test]
    fn toggle_current_field_blocks_when_settings_load_failed() {
        let mut screen = SettingsScreen::new();
        let mut config = Config::default();
        screen.settings_load_error = Some("bad json".to_string());
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 0;
        screen.auto_compact_enabled = false;

        toggle_current_field(&mut screen, &mut config);

        assert!(!screen.auto_compact_enabled);
        assert!(screen
            .last_status_message
            .as_deref()
            .unwrap_or_default()
            .contains("Settings not saved"));
    }

    #[test]
    fn toggle_current_field_general_field1_flips_notifications() {
        let (mut screen, _settings_file) =
            settings_screen_with_temp_settings(r#"{"notifications":true}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 5; // Desktop notifications

        let before = screen.notifications_enabled;
        toggle_current_field(&mut screen, &mut config);
        assert_eq!(screen.notifications_enabled, !before);
    }

    #[test]
    fn toggle_current_field_display_field0_flips_reduce_motion() {
        let (mut screen, _settings_file) =
            settings_screen_with_temp_settings(r#"{"reduceMotion":false}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::Display;
        screen.selected_field = 2; // Reduce motion

        let before = screen.reduce_motion;
        toggle_current_field(&mut screen, &mut config);
        assert_eq!(screen.reduce_motion, !before);
    }

    #[test]
    fn toggle_current_field_display_field1_flips_terminal_progress_bar() {
        let (mut screen, _settings_file) =
            settings_screen_with_temp_settings(r#"{"terminalProgressBar":true}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::Display;
        screen.selected_field = 3; // Terminal progress bar

        let before = screen.terminal_progress_bar;
        toggle_current_field(&mut screen, &mut config);
        assert_eq!(screen.terminal_progress_bar, !before);
    }

    // ---------------------------------------------------------------------------
    // General tab render includes auto-compact row
    // ---------------------------------------------------------------------------

    #[test]
    fn general_tab_contains_auto_compact_row() {
        let mut screen = SettingsScreen::new();
        screen.auto_compact_enabled = true;
        screen.auto_compact_threshold = 95;

        let lines = build_general_lines(&screen);

        // Flatten all spans to a single string for easy assertion
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();

        assert!(
            text.contains("Auto-compact"),
            "General tab should render an Auto-compact row; got: {}",
            mangocode_core::truncate::truncate_bytes_prefix(&text, 300)
        );
        assert!(
            text.contains("95%"),
            "General tab should show the threshold percentage in the threshold field"
        );
    }

    #[test]
    fn general_tab_contains_notifications_row() {
        let screen = SettingsScreen::new();
        let lines = build_general_lines(&screen);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            text.contains("Desktop notifications"),
            "General tab should render a Notifications row"
        );
    }

    #[test]
    fn display_tab_contains_reduce_motion_and_progress_bar_rows() {
        let screen = SettingsScreen::new();
        let lines = build_display_lines(&screen);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            text.contains("Reduce motion"),
            "Display tab should have Reduce motion row"
        );
        assert!(
            text.contains("Terminal progress bar"),
            "Display tab should have Terminal progress bar row"
        );
    }

    #[test]
    fn output_style_lists_include_project_styles() {
        let project_dir = TempDir::new().unwrap();
        let styles_dir = project_dir.path().join(".mangocode").join("output-styles");
        std::fs::create_dir_all(&styles_dir).unwrap();
        std::fs::write(
            styles_dir.join("project-ui.md"),
            "# Project UI\nProject display style.\n\nUse project UI style.",
        )
        .unwrap();

        let mut screen = SettingsScreen::new();
        screen.settings_snapshot.config.project_dir = Some(project_dir.path().to_path_buf());

        let lines = build_display_lines(&screen);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();

        assert!(text.contains("project-ui"));
    }

    // ---------------------------------------------------------------------------
    // max_fields_for_tab sanity
    // ---------------------------------------------------------------------------

    #[test]
    fn max_fields_for_tab_returns_correct_counts() {
        let mut screen = SettingsScreen::new();

        screen.active_tab = SettingsTab::General;
        assert_eq!(screen.max_fields_for_tab(), 7);

        screen.active_tab = SettingsTab::Display;
        assert_eq!(screen.max_fields_for_tab(), 4);

        screen.active_tab = SettingsTab::Privacy;
        assert_eq!(screen.max_fields_for_tab(), 2);

        screen.active_tab = SettingsTab::Advanced;
        assert_eq!(screen.max_fields_for_tab(), 0);

        screen.active_tab = SettingsTab::KeyBindings;
        assert_eq!(screen.max_fields_for_tab(), 0);
    }

    // ---------------------------------------------------------------------------
    // Unified navigation / activation
    // ---------------------------------------------------------------------------

    #[test]
    fn navigation_reaches_every_general_field() {
        let mut screen = SettingsScreen::new();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 0;
        let count = screen.field_count();
        assert_eq!(count, 7);

        for _ in 0..20 {
            screen.select_next();
        }
        assert_eq!(screen.selected_field, count - 1, "Down should reach last field");

        for _ in 0..20 {
            screen.select_prev();
        }
        assert_eq!(screen.selected_field, 0, "Up should reach first field");
    }

    #[test]
    fn informational_tab_scrolls_instead_of_selecting() {
        let mut screen = SettingsScreen::new();
        screen.active_tab = SettingsTab::Advanced;
        assert_eq!(screen.field_count(), 0);
        screen.scroll_offset = 0;

        screen.select_next();
        assert_eq!(screen.scroll_offset, 1, "Down scrolls on informational tabs");
        screen.select_prev();
        assert_eq!(screen.scroll_offset, 0, "Up scrolls on informational tabs");
    }

    #[test]
    fn enter_on_text_field_begins_edit() {
        let (mut screen, _file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 0; // Model

        activate_current_field(&mut screen, &mut config, 1, true);

        assert_eq!(screen.edit_field.as_deref(), Some("model"));
    }

    #[test]
    fn arrow_keys_do_not_open_text_editor() {
        let (mut screen, _file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 0; // Model

        activate_current_field(&mut screen, &mut config, 1, false); // ←/→/Space

        assert!(
            screen.edit_field.is_none(),
            "directional keys must not open the inline editor"
        );
    }

    #[test]
    fn cycle_output_format_wraps_backwards() {
        let (mut screen, _file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default(); // output_format = Text (index 0)
        screen.active_tab = SettingsTab::Display;
        screen.selected_field = 0; // Output Format

        activate_current_field(&mut screen, &mut config, -1, false); // wrap to last

        assert!(matches!(
            config.output_format,
            mangocode_core::config::OutputFormat::StreamJson
        ));
    }

    #[test]
    fn telemetry_toggle_writes_disable_telemetry_inverted() {
        let (mut screen, settings_file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::Privacy;
        screen.selected_field = 0; // Telemetry
        screen.telemetry_enabled = false;

        activate_current_field(&mut screen, &mut config, 1, false);

        assert!(screen.telemetry_enabled, "telemetry should turn on");
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(settings_file.path()).unwrap()).unwrap();
        assert_eq!(
            value["disableTelemetry"], false,
            "telemetry on → disableTelemetry false"
        );
    }

    #[test]
    fn verbose_toggle_persists_to_config() {
        let (mut screen, settings_file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::Display;
        screen.selected_field = 1; // Verbose Mode

        activate_current_field(&mut screen, &mut config, 1, false);

        assert!(config.verbose, "verbose should turn on");
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(settings_file.path()).unwrap()).unwrap();
        assert_eq!(value["config"]["verbose"], true);
    }

    #[test]
    fn auto_compact_toggle_sets_config_field() {
        // Start from an explicitly-disabled state so the toggle deterministically
        // enables it (the default is now on).
        let (mut screen, settings_file) =
            settings_screen_with_temp_settings(r#"{"config":{"auto_compact":false}}"#);
        let mut config = Config {
            auto_compact: false,
            ..Default::default()
        };
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 3; // Auto-compact
        assert!(!screen.auto_compact_enabled, "precondition: starts disabled");

        activate_current_field(&mut screen, &mut config, 1, false);

        assert!(config.auto_compact, "toggling should enable config.auto_compact");
        assert!(screen.auto_compact_enabled);
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(settings_file.path()).unwrap()).unwrap();
        assert_eq!(value["config"]["auto_compact"], true);
    }

    #[test]
    fn threshold_edit_sets_compact_threshold() {
        let (mut screen, _file) = settings_screen_with_temp_settings(r#"{}"#);
        let mut config = Config::default();
        screen.active_tab = SettingsTab::General;
        screen.selected_field = 4; // Auto-compact threshold (Text)

        // Enter begins editing the threshold field.
        activate_current_field(&mut screen, &mut config, 1, true);
        assert_eq!(screen.edit_field.as_deref(), Some("compact_threshold"));

        screen.edit_value = "75".to_string();
        screen.commit_edit();
        screen.apply_and_save(&mut config).unwrap();

        assert!(
            (config.compact_threshold - 0.75).abs() < 1e-6,
            "compact_threshold should be 0.75, got {}",
            config.compact_threshold
        );
    }
}
