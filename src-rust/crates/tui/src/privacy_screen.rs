// privacy_screen.rs — Privacy settings dialog.
//
// A focused overlay with toggle-style controls for privacy preferences.
// Opened by /privacy. Changes are persisted to privacy fields in settings.json.

use mangocode_core::config::Settings;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use serde_json::Value;
use std::path::Path;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// State for a single privacy toggle.
#[derive(Debug, Clone)]
pub struct PrivacyToggle {
    pub key: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub enabled: bool,
}

pub struct PrivacyScreen {
    pub visible: bool,
    pub toggles: Vec<PrivacyToggle>,
    pub selected_idx: usize,
    pub settings_load_error: Option<String>,
    pub last_status_message: Option<String>,
}

impl PrivacyScreen {
    pub fn new() -> Self {
        Self {
            visible: false,
            toggles: default_toggles(),
            selected_idx: 0,
            settings_load_error: None,
            last_status_message: None,
        }
    }

    pub fn open(&mut self) {
        self.visible = true;
        self.selected_idx = 0;
        let (toggles, load_error) = load_privacy_toggles();
        self.toggles = toggles;
        self.settings_load_error = load_error;
        self.last_status_message = self
            .settings_load_error
            .as_deref()
            .map(privacy_load_warning_message);
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn select_prev(&mut self) {
        if self.selected_idx > 0 {
            self.selected_idx -= 1;
        }
    }

    pub fn select_next(&mut self) {
        if self.selected_idx + 1 < self.toggles.len() {
            self.selected_idx += 1;
        }
    }

    /// Toggle the currently selected privacy option.
    pub fn toggle_selected(&mut self) {
        if let Err(err) = self.try_toggle_selected() {
            self.last_status_message = Some(err);
        }
    }

    pub fn take_status_message(&mut self) -> Option<String> {
        self.last_status_message.take()
    }

    fn try_toggle_selected(&mut self) -> Result<(), String> {
        if let Some(load_error) = self.settings_load_error.as_deref() {
            return Err(privacy_save_blocked_message(load_error));
        }

        if self.selected_idx >= self.toggles.len() {
            return Ok(());
        }

        let previous = self.toggles[self.selected_idx].enabled;
        self.toggles[self.selected_idx].enabled = !previous;
        match self.save() {
            Ok(()) => {
                self.last_status_message = None;
                Ok(())
            }
            Err(err) => {
                self.toggles[self.selected_idx].enabled = previous;
                Err(format!("Privacy settings not saved: {err}"))
            }
        }
    }

    /// Persist the current toggle state to settings.
    pub fn save(&self) -> Result<(), String> {
        save_privacy_toggles_to_path(&Settings::global_settings_path(), &self.toggles)
    }
}

impl Default for PrivacyScreen {
    fn default() -> Self {
        Self::new()
    }
}

fn privacy_load_warning_message(error: &str) -> String {
    format!("Privacy settings could not be loaded: {error}")
}

fn privacy_save_blocked_message(error: &str) -> String {
    format!("Privacy settings not saved because settings.json could not be loaded: {error}")
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

fn apply_settings_json_to_toggles(value: &Value, toggles: &mut [PrivacyToggle]) {
    for toggle in toggles {
        match toggle.key {
            "telemetry" => {
                if let Some(disabled) = value.get("disableTelemetry").and_then(Value::as_bool) {
                    toggle.enabled = !disabled;
                }
            }
            "usage_sharing" => {
                if let Some(enabled) = value.get("shareUsageData").and_then(Value::as_bool) {
                    toggle.enabled = enabled;
                }
            }
            "api_request_logging" => {
                if let Some(enabled) = value.get("apiRequestLogging").and_then(Value::as_bool) {
                    toggle.enabled = enabled;
                }
            }
            "error_reporting" => {
                if let Some(enabled) = value.get("errorReporting").and_then(Value::as_bool) {
                    toggle.enabled = enabled;
                }
            }
            _ => {}
        }
    }
}

fn apply_toggles_to_settings_json(
    value: &mut Value,
    toggles: &[PrivacyToggle],
) -> Result<(), String> {
    let Some(obj) = value.as_object_mut() else {
        return Err("settings.json must contain a JSON object".to_string());
    };

    for toggle in toggles {
        match toggle.key {
            "telemetry" => {
                obj.insert("disableTelemetry".to_string(), Value::Bool(!toggle.enabled));
            }
            "usage_sharing" => {
                obj.insert("shareUsageData".to_string(), Value::Bool(toggle.enabled));
            }
            "api_request_logging" => {
                obj.insert("apiRequestLogging".to_string(), Value::Bool(toggle.enabled));
            }
            "error_reporting" => {
                obj.insert("errorReporting".to_string(), Value::Bool(toggle.enabled));
            }
            _ => {}
        }
    }

    Ok(())
}

fn load_privacy_toggles() -> (Vec<PrivacyToggle>, Option<String>) {
    let path = Settings::global_settings_path();
    match load_privacy_toggles_from_path(&path) {
        Ok(toggles) => (toggles, None),
        Err(err) => (default_toggles(), Some(err)),
    }
}

fn load_privacy_toggles_from_path(path: &Path) -> Result<Vec<PrivacyToggle>, String> {
    let mut toggles = default_toggles();
    let value = read_settings_json_value(path)?;
    if !value.is_object() {
        return Err(format!("{} must contain a JSON object", path.display()));
    }
    apply_settings_json_to_toggles(&value, &mut toggles);
    Ok(toggles)
}

fn save_privacy_toggles_to_path(path: &Path, toggles: &[PrivacyToggle]) -> Result<(), String> {
    let mut value = read_settings_json_value(path)?;
    apply_toggles_to_settings_json(&mut value, toggles)?;
    write_settings_json_value(path, &value)
}

fn default_toggles() -> Vec<PrivacyToggle> {
    vec![
        PrivacyToggle {
            key: "telemetry",
            label: "Telemetry",
            description: "Send anonymised crash reports and usage statistics to Anthropic \
                          to help diagnose issues and improve the product.",
            enabled: false,
        },
        PrivacyToggle {
            key: "usage_sharing",
            label: "Usage Sharing",
            description: "Share aggregate usage patterns (no personal data) to help \
                          Anthropic understand how MangoCode is used.",
            enabled: false,
        },
        PrivacyToggle {
            key: "api_request_logging",
            label: "API Request Logging",
            description: "Log API requests to a local file (~/.mangocode/api_requests.log) \
                          for debugging. Logs are stored locally only.",
            enabled: false,
        },
        PrivacyToggle {
            key: "error_reporting",
            label: "Error Reporting",
            description: "Automatically report errors and stack traces to Anthropic. \
                          Helps fix bugs faster. No conversation content is included.",
            enabled: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the privacy settings dialog.
pub fn render_privacy_screen(frame: &mut Frame, screen: &PrivacyScreen, area: Rect) {
    if !screen.visible {
        return;
    }

    let dialog_width = 68u16.min(area.width.saturating_sub(4));
    // Each toggle: 1 label line + 2 description lines + 1 spacer = 4 rows
    let toggle_rows = (screen.toggles.len() as u16) * 4;
    let dialog_height = (toggle_rows + 8).min(area.height.saturating_sub(4));
    let dialog_area = crate::overlays::centered_rect(dialog_width, dialog_height, area);

    frame.render_widget(Clear, dialog_area);

    let mut lines: Vec<Line> = vec![
        Line::from(vec![Span::styled(
            "  Privacy Settings",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            if screen.settings_load_error.is_some() {
                "  Fix settings.json before changing privacy preferences."
            } else {
                "  Use Space or Enter to toggle. Changes are saved immediately."
            },
            Style::default()
                .fg(if screen.settings_load_error.is_some() {
                    Color::Yellow
                } else {
                    Color::DarkGray
                })
                .add_modifier(Modifier::ITALIC),
        )]),
        Line::from(""),
    ];

    if let Some(load_error) = screen.settings_load_error.as_deref() {
        lines.push(Line::from(vec![Span::styled(
            format!("  {load_error}"),
            Style::default().fg(Color::LightRed),
        )]));
        lines.push(Line::from(""));
    }

    for (i, toggle) in screen.toggles.iter().enumerate() {
        let is_selected = i == screen.selected_idx;

        let prefix = if is_selected { "  \u{25BA} " } else { "    " };

        let (toggle_text, toggle_fg) = if toggle.enabled {
            ("[ ON  ]", Color::Green)
        } else {
            ("[ OFF ]", Color::Red)
        };

        let label_style = if is_selected {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        // Label row with toggle
        lines.push(Line::from(vec![
            Span::raw(prefix),
            Span::styled(format!("{:<28}", toggle.label), label_style),
            Span::styled(
                toggle_text.to_string(),
                Style::default().fg(toggle_fg).add_modifier(Modifier::BOLD),
            ),
        ]));

        // Description (wrapped to fit dialog width - 8 chars for indentation)
        let desc_max = (dialog_width as usize).saturating_sub(10);
        let wrapped = word_wrap_str(toggle.description, desc_max);
        for desc_line in &wrapped {
            lines.push(Line::from(vec![Span::styled(
                format!("       {}", desc_line),
                Style::default().fg(Color::DarkGray),
            )]));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(vec![Span::styled(
        "  \u{2191}\u{2193} navigate  \u{00b7}  Space/Enter toggle  \u{00b7}  Esc close",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    )]));

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Privacy Settings ")
        .border_style(Style::default().fg(Color::Cyan));

    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, dialog_area);
}

fn word_wrap_str(text: &str, width: usize) -> Vec<String> {
    if width == 0 || text.len() <= width {
        return vec![text.to_string()];
    }
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for word in text.split_whitespace() {
        let ww = word.len();
        if current_width == 0 {
            current.push_str(word);
            current_width = ww;
        } else if current_width + 1 + ww <= width {
            current.push(' ');
            current.push_str(word);
            current_width += 1 + ww;
        } else {
            result.push(std::mem::take(&mut current));
            current.push_str(word);
            current_width = ww;
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    if result.is_empty() {
        result.push(text.to_string());
    }
    result
}

// ---------------------------------------------------------------------------
// Key handling helpers (called from app.rs)
// ---------------------------------------------------------------------------

/// Returns `true` if the key event was consumed by the privacy screen.
pub fn handle_privacy_key(screen: &mut PrivacyScreen, key: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyCode;

    if !screen.visible {
        return false;
    }

    match key.code {
        KeyCode::Esc => {
            screen.close();
        }
        KeyCode::Up => {
            screen.select_prev();
        }
        KeyCode::Down => {
            screen.select_next();
        }
        KeyCode::Enter | KeyCode::Char(' ') => {
            screen.toggle_selected();
        }
        _ => {}
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_privacy_toggles_reads_settings_json_keys() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{
                "disableTelemetry": false,
                "shareUsageData": true,
                "apiRequestLogging": true,
                "errorReporting": false
            }"#,
        )
        .unwrap();

        let toggles = load_privacy_toggles_from_path(&path).unwrap();

        assert!(
            toggles
                .iter()
                .find(|t| t.key == "telemetry")
                .unwrap()
                .enabled
        );
        assert!(
            toggles
                .iter()
                .find(|t| t.key == "usage_sharing")
                .unwrap()
                .enabled
        );
        assert!(
            toggles
                .iter()
                .find(|t| t.key == "api_request_logging")
                .unwrap()
                .enabled
        );
        assert!(
            !toggles
                .iter()
                .find(|t| t.key == "error_reporting")
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn save_privacy_toggles_preserves_unknown_settings() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{
                "unknownTopLevel": "keep",
                "config": {
                    "model": "openai/gpt-4o"
                }
            }"#,
        )
        .unwrap();
        let mut toggles = default_toggles();
        for toggle in &mut toggles {
            toggle.enabled = matches!(toggle.key, "telemetry" | "usage_sharing");
        }

        save_privacy_toggles_to_path(&path, &toggles).unwrap();

        let value: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["unknownTopLevel"], "keep");
        assert_eq!(value["config"]["model"], "openai/gpt-4o");
        assert_eq!(value["disableTelemetry"], false);
        assert_eq!(value["shareUsageData"], true);
        assert_eq!(value["apiRequestLogging"], false);
        assert_eq!(value["errorReporting"], false);
    }

    #[test]
    fn toggle_selected_blocks_when_settings_load_failed() {
        let mut screen = PrivacyScreen::new();
        screen.settings_load_error = Some("bad json".to_string());
        screen.selected_idx = 0;
        screen.toggles[0].enabled = false;

        screen.toggle_selected();

        assert!(!screen.toggles[0].enabled);
        assert!(screen
            .last_status_message
            .as_deref()
            .unwrap_or_default()
            .contains("Privacy settings not saved"));
    }
}
