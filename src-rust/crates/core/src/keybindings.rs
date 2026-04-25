//! Configurable keyboard shortcuts system

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// All keybinding contexts
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum KeyContext {
    Global,
    Chat,
    Autocomplete,
    Confirmation,
    Help,
    Transcript,
    HistorySearch,
    Task,
    ThemePicker,
    Settings,
    Tabs,
    Attachments,
    Footer,
    MessageSelector,
    DiffDialog,
    ModelPicker,
    Select,
    Plugin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedKeystroke {
    pub key: String, // normalized key name
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
}

pub type Chord = Vec<ParsedKeystroke>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedBinding {
    pub chord: Chord,
    pub action: Option<String>, // None = unbound
    pub context: KeyContext,
}

/// Parse a keystroke string like "ctrl+shift+enter" into ParsedKeystroke
pub fn parse_keystroke(s: &str) -> Option<ParsedKeystroke> {
    let s = s.trim().to_lowercase();
    let mut ctrl = false;
    let mut alt = false;
    let mut shift = false;
    let mut meta = false;
    let mut key_parts: Vec<&str> = Vec::new();

    for part in s.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part {
            "ctrl" | "control" => ctrl = true,
            "alt" | "opt" | "option" => alt = true,
            "shift" => shift = true,
            "meta" | "cmd" | "command" | "super" | "win" => meta = true,
            _ => key_parts.push(part),
        }
    }

    if key_parts.is_empty() {
        return None;
    }

    let key = normalize_key(key_parts.join("+").as_str());
    Some(ParsedKeystroke {
        key,
        ctrl,
        alt,
        shift,
        meta,
    })
}

fn normalize_key(k: &str) -> String {
    match k {
        "esc" | "escape" => "escape".to_string(),
        "return" | "enter" => "enter".to_string(),
        "del" | "delete" => "delete".to_string(),
        "backspace" | "bs" => "backspace".to_string(),
        "space" | " " => "space".to_string(),
        "up" => "up".to_string(),
        "down" => "down".to_string(),
        "left" => "left".to_string(),
        "right" => "right".to_string(),
        "pageup" | "pgup" => "pageup".to_string(),
        "pagedown" | "pgdn" | "pgdown" => "pagedown".to_string(),
        "home" => "home".to_string(),
        "end" => "end".to_string(),
        "tab" => "tab".to_string(),
        "f1" => "f1".to_string(),
        k => k.to_string(),
    }
}

/// Parse a chord (space-separated keystrokes like "ctrl+k ctrl+d")
pub fn parse_chord(s: &str) -> Option<Chord> {
    let keystrokes: Vec<ParsedKeystroke> =
        s.split_whitespace().filter_map(parse_keystroke).collect();
    if keystrokes.is_empty() {
        None
    } else {
        Some(keystrokes)
    }
}

/// Keys that cannot be rebound
pub const NON_REBINDABLE: &[&str] = &["ctrl+c", "ctrl+d", "ctrl+m"];

/// Keybinding profile for different terminal environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeybindingProfile {
    /// Standard profile for normal terminals (full keybinding set).
    Standard,
    /// IDE-compatible profile with safer fallback chords for IDE terminals.
    IdeCompatible,
}

/// Default keybindings with comprehensive coverage of text editing, navigation, vim, and TUI actions
///
/// # Standard Keybindings (Phase 1 Implementation)
/// - **Ctrl+L**: Clear current input line (like bash) [Chat context only due to conflict]
/// - **Ctrl+U**: Kill input from cursor to start of line (Emacs-style)
/// - **Alt+←/Alt+→**: Navigate to previous/next message in transcript
/// - **Ctrl+. (Ctrl+>)**: Jump to next error/issue in messages
/// - **Ctrl+Shift+.**: Jump to previous error/issue
/// - **Ctrl+M**: Send message (alternative to Enter)
/// - **Shift+Tab**: Reverse indent/unindent in input (cycle permission mode)
/// - **Ctrl+H**: Delete character before cursor (Chat context, Emacs-style)
/// - **Alt+H**: Open help (alternative to F1)
/// - **Ctrl+O**: Expand/collapse truncated tool output when applicable; otherwise previous history entry
/// - **Ctrl+I**: Jump forward in history
/// - **Alt+D**: Delete word forward (already implemented)
/// - **Ctrl+V**: Paste from clipboard (already implemented)
pub fn default_bindings() -> Vec<ParsedBinding> {
    let defaults: &[(&str, &str, KeyContext)] = &[
        // ========== GLOBAL CONTROL ==========
        ("ctrl+c", "session.abort", KeyContext::Global),
        ("ctrl+d", "session.close", KeyContext::Global),
        ("ctrl+l", "view.repaint", KeyContext::Global),
        ("ctrl+r", "history.overlay.open", KeyContext::Global),
        ("ctrl+b", "createBranch", KeyContext::Global),
        ("alt+h", "openHelp", KeyContext::Global),
        // ========== CHAT / INPUT CONTEXT ==========
        ("enter", "chat.submit", KeyContext::Chat),
        ("up", "history.entry.prev", KeyContext::Chat),
        ("down", "history.entry.next", KeyContext::Chat),
        ("shift+tab", "mode.cycle", KeyContext::Chat),
        ("pageup", "transcript.scroll.up", KeyContext::Chat),
        ("pagedown", "transcript.scroll.down", KeyContext::Chat),
        ("tab", "suggestion.accept", KeyContext::Chat),
        ("shift+enter", "newline", KeyContext::Chat),
        ("home", "goLineStart", KeyContext::Chat),
        ("end", "goLineEnd", KeyContext::Chat),
        // Text Editing (Emacs-style)
        ("ctrl+a", "goLineStart", KeyContext::Chat),
        ("ctrl+e", "goLineEnd", KeyContext::Chat),
        ("ctrl+h", "deleteCharBefore", KeyContext::Chat),
        ("ctrl+k", "killToEnd", KeyContext::Chat),
        ("ctrl+u", "killToStart", KeyContext::Chat),
        ("ctrl+w", "killWord", KeyContext::Chat),
        ("alt+d", "deleteWord", KeyContext::Chat),
        ("alt+backspace", "killWord", KeyContext::Chat),
        // New Text Editing & Navigation
        ("ctrl+m", "chat.submit.alt", KeyContext::Chat),
        ("ctrl+l", "input.clear", KeyContext::Chat),
        ("ctrl+.", "transcript.issue.next", KeyContext::Chat),
        ("ctrl+shift+.", "transcript.issue.prev", KeyContext::Chat),
        ("alt+left", "transcript.jump.prev", KeyContext::Chat),
        ("alt+right", "transcript.jump.next", KeyContext::Chat),
        // Ctrl+O is handled first in the TUI for tool-output expand; if nothing matches, history prev applies.
        ("ctrl+o", "history.entry.prev", KeyContext::Chat),
        ("ctrl+i", "history.entry.next", KeyContext::Chat),
        // Searching
        ("ctrl+f", "search.transcript.open", KeyContext::Chat),
        ("ctrl+shift+f", "search.global.open", KeyContext::Chat),
        ("ctrl+g", "goToLine", KeyContext::Chat),
        ("f3", "search.transcript.next", KeyContext::Chat),
        ("ctrl+]", "search.transcript.next", KeyContext::Chat),
        ("shift+f3", "search.transcript.prev", KeyContext::Chat),
        ("ctrl+[", "search.transcript.prev", KeyContext::Chat),
        // ========== CONFIRMATION DIALOGS ==========
        ("y", "confirm.accept", KeyContext::Confirmation),
        ("enter", "confirm.accept", KeyContext::Confirmation),
        ("n", "confirm.reject", KeyContext::Confirmation),
        ("escape", "confirm.reject", KeyContext::Confirmation),
        ("up", "confirm.prev", KeyContext::Confirmation),
        ("down", "confirm.next", KeyContext::Confirmation),
        // ========== HELP OVERLAY ==========
        ("escape", "overlay.dismiss", KeyContext::Help),
        ("q", "overlay.dismiss", KeyContext::Help),
        ("up", "scrollUp", KeyContext::Help),
        ("down", "scrollDown", KeyContext::Help),
        ("pageup", "pageUp", KeyContext::Help),
        ("pagedown", "pageDown", KeyContext::Help),
        // ========== HISTORY SEARCH ==========
        ("enter", "history.result.select", KeyContext::HistorySearch),
        ("escape", "history.result.cancel", KeyContext::HistorySearch),
        ("up", "history.result.prev", KeyContext::HistorySearch),
        ("down", "history.result.next", KeyContext::HistorySearch),
        ("tab", "togglePreview", KeyContext::HistorySearch),
        // ========== TRANSCRIPT / MESSAGE SELECTION ==========
        ("up", "prevMessage", KeyContext::Transcript),
        ("down", "nextMessage", KeyContext::Transcript),
        ("pageup", "pageUp", KeyContext::Transcript),
        ("pagedown", "pageDown", KeyContext::Transcript),
        ("home", "goStart", KeyContext::Transcript),
        ("end", "goEnd", KeyContext::Transcript),
        ("enter", "selectMessage", KeyContext::Transcript),
        ("escape", "cancel", KeyContext::Transcript),
        // ========== MESSAGE SELECTOR OVERLAY ==========
        ("up", "prevMessage", KeyContext::MessageSelector),
        ("down", "nextMessage", KeyContext::MessageSelector),
        ("enter", "select", KeyContext::MessageSelector),
        ("escape", "cancel", KeyContext::MessageSelector),
        ("j", "nextMessage", KeyContext::MessageSelector),
        ("k", "prevMessage", KeyContext::MessageSelector),
        // ========== THEME & MODEL PICKERS ==========
        ("up", "prev", KeyContext::ThemePicker),
        ("down", "next", KeyContext::ThemePicker),
        ("pageup", "pageUp", KeyContext::ThemePicker),
        ("pagedown", "pageDown", KeyContext::ThemePicker),
        ("enter", "select", KeyContext::ThemePicker),
        ("escape", "cancel", KeyContext::ThemePicker),
        ("j", "next", KeyContext::ThemePicker),
        ("k", "prev", KeyContext::ThemePicker),
        // ========== TASK LIST ==========
        ("up", "prevTask", KeyContext::Task),
        ("down", "nextTask", KeyContext::Task),
        ("enter", "selectTask", KeyContext::Task),
        ("escape", "closeTask", KeyContext::Task),
        ("x", "toggleDone", KeyContext::Task),
        // ========== DIFF DIALOG ==========
        ("up", "prevDiff", KeyContext::DiffDialog),
        ("down", "nextDiff", KeyContext::DiffDialog),
        ("pageup", "pageUp", KeyContext::DiffDialog),
        ("pagedown", "pageDown", KeyContext::DiffDialog),
        ("enter", "acceptDiff", KeyContext::DiffDialog),
        ("escape", "rejectDiff", KeyContext::DiffDialog),
        ("r", "rejectDiff", KeyContext::DiffDialog),
        ("a", "acceptDiff", KeyContext::DiffDialog),
        // ========== MODAL SELECT (Generic) ==========
        ("up", "prev", KeyContext::Select),
        ("down", "next", KeyContext::Select),
        ("pageup", "pageUp", KeyContext::Select),
        ("pagedown", "pageDown", KeyContext::Select),
        ("enter", "select", KeyContext::Select),
        ("escape", "cancel", KeyContext::Select),
        ("j", "next", KeyContext::Select),
        ("k", "prev", KeyContext::Select),
        ("/", "search", KeyContext::Select),
        // ========== PLUGIN & ATTACHMENTS ==========
        ("up", "prev", KeyContext::Plugin),
        ("down", "next", KeyContext::Plugin),
        ("enter", "select", KeyContext::Plugin),
        ("escape", "cancel", KeyContext::Plugin),
        ("space", "toggle", KeyContext::Attachments),
        ("a", "addAttachment", KeyContext::Attachments),
        ("r", "removeAttachment", KeyContext::Attachments),
    ];

    defaults
        .iter()
        .filter_map(|(chord_str, action, context)| {
            parse_chord(chord_str).map(|chord| ParsedBinding {
                chord,
                action: Some(action.to_string()),
                context: context.clone(),
            })
        })
        .collect()
}

/// IDE-compatible keybindings with safer fallback chords.
///
/// These bindings add Ctrl+Q prefix chords for actions that conflict
/// with IDE shortcuts (Ctrl+R, Alt+H, Ctrl+F, etc.).
/// The original bindings are kept for compatibility, but the IDE-safe
/// chords are preferred in help text and documentation.
pub fn ide_bindings() -> Vec<ParsedBinding> {
    let ide_fallbacks: &[(&str, &str, KeyContext)] = &[
        // IDE-safe fallback chords using Ctrl+Q prefix (Ctrl+K conflicts with IDE shortcuts)
        ("ctrl+q r", "history.overlay.open", KeyContext::Global),
        ("ctrl+q h", "openHelp", KeyContext::Global),
        ("ctrl+q f1", "openHelp", KeyContext::Global),
        ("ctrl+q a", "goLineStart", KeyContext::Chat),
        ("ctrl+q left", "transcript.jump.prev", KeyContext::Chat),
        ("ctrl+q right", "transcript.jump.next", KeyContext::Chat),
        ("ctrl+q f", "search.transcript.open", KeyContext::Chat),
        ("ctrl+q shift+f", "search.global.open", KeyContext::Chat),
        ("ctrl+q g", "goToLine", KeyContext::Chat),
        ("ctrl+q ]", "search.transcript.next", KeyContext::Chat),
        ("ctrl+q [", "search.transcript.prev", KeyContext::Chat),
    ];

    ide_fallbacks
        .iter()
        .filter_map(|(chord_str, action, context)| {
            parse_chord(chord_str).map(|chord| ParsedBinding {
                chord,
                action: Some(action.to_string()),
                context: context.clone(),
            })
        })
        .collect()
}

/// Get bindings for a specific keybinding profile.
///
/// - `Standard`: Returns the default bindings only.
/// - `IdeCompatible`: Returns default bindings plus IDE-safe fallback chords.
pub fn bindings_for_profile(profile: KeybindingProfile) -> Vec<ParsedBinding> {
    let mut bindings = default_bindings();
    if profile == KeybindingProfile::IdeCompatible {
        bindings.extend(ide_bindings());
    }
    bindings
}

/// User keybindings loaded from ~/.mangocode/keybindings.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserKeybindings {
    pub bindings: Vec<UserBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonKeybindingConfig {
    #[serde(default)]
    bindings: Vec<JsonKeybindingBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonKeybindingBlock {
    context: String,
    bindings: IndexMap<String, Option<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBinding {
    pub chord: String,          // e.g. "ctrl+k ctrl+d"
    pub action: Option<String>, // None = unbound
    pub context: Option<String>,
}

impl UserKeybindings {
    pub fn from_json_str(content: &str) -> Self {
        serde_json::from_str(content)
            .or_else(|_| Self::from_block_config(content))
            .unwrap_or_default()
    }

    pub fn load(config_dir: &Path) -> Self {
        let path = config_dir.join("keybindings.json");
        if let Ok(content) = std::fs::read_to_string(&path) {
            Self::from_json_str(&content)
        } else {
            Self::default()
        }
    }

    pub fn save(&self, config_dir: &Path) -> anyhow::Result<()> {
        let path = config_dir.join("keybindings.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn from_block_config(content: &str) -> Result<Self, serde_json::Error> {
        let config: JsonKeybindingConfig = serde_json::from_str(content)?;
        let bindings = config
            .bindings
            .into_iter()
            .flat_map(|block| {
                let context = block.context;
                block
                    .bindings
                    .into_iter()
                    .map(move |(chord, action)| UserBinding {
                        chord,
                        action,
                        context: Some(context.clone()),
                    })
            })
            .collect();
        Ok(Self { bindings })
    }
}

/// Resolved keybindings (defaults merged with user overrides)
pub struct KeybindingResolver {
    bindings: Vec<ParsedBinding>,
    pending_chord: Vec<ParsedKeystroke>,
}

impl KeybindingResolver {
    pub fn new(user: &UserKeybindings) -> Self {
        Self::with_profile(user, KeybindingProfile::Standard)
    }

    pub fn with_profile(user: &UserKeybindings, profile: KeybindingProfile) -> Self {
        let mut bindings = bindings_for_profile(profile);

        // Apply user overrides (user bindings win, last match wins)
        for user_binding in &user.bindings {
            if let Some(chord) = parse_chord(&user_binding.chord) {
                let context = user_binding
                    .context
                    .as_deref()
                    .and_then(|c| serde_json::from_str(&format!("\"{}\"", c)).ok())
                    .unwrap_or(KeyContext::Global);

                bindings.push(ParsedBinding {
                    chord,
                    action: user_binding.action.clone(),
                    context,
                });
            }
        }

        Self {
            bindings,
            pending_chord: Vec::new(),
        }
    }

    /// Process a keystroke, returns action if binding matches
    pub fn process(
        &mut self,
        keystroke: ParsedKeystroke,
        context: &KeyContext,
    ) -> KeybindingResult {
        self.pending_chord.push(keystroke);

        // Find matching bindings in current context + Global
        let matches: Vec<&ParsedBinding> = self
            .bindings
            .iter()
            .filter(|b| &b.context == context || b.context == KeyContext::Global)
            .filter(|b| b.chord.starts_with(self.pending_chord.as_slice()))
            .collect();

        if matches.is_empty() {
            self.pending_chord.clear();
            return KeybindingResult::NoMatch;
        }

        let exact: Vec<&ParsedBinding> = matches
            .iter()
            .copied()
            .filter(|b| b.chord.len() == self.pending_chord.len())
            .collect();

        if !exact.is_empty() {
            // Last match wins (user overrides)
            let Some(binding) = exact.last() else {
                self.pending_chord.clear();
                return KeybindingResult::NoMatch;
            };
            self.pending_chord.clear();
            return match &binding.action {
                Some(action) => KeybindingResult::Action(action.clone()),
                None => KeybindingResult::Unbound,
            };
        }

        // Chord in progress
        KeybindingResult::Pending
    }

    pub fn cancel_chord(&mut self) {
        self.pending_chord.clear();
    }

    pub fn has_pending_chord(&self) -> bool {
        !self.pending_chord.is_empty()
    }
}

impl PartialEq for ParsedKeystroke {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.ctrl == other.ctrl
            && self.alt == other.alt
            && self.shift == other.shift
            && self.meta == other.meta
    }
}

#[derive(Debug, Clone)]
pub enum KeybindingResult {
    Action(String),
    Unbound,
    Pending,
    NoMatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_keystroke_simple() {
        let ks = parse_keystroke("enter").unwrap();
        assert_eq!(ks.key, "enter");
        assert!(!ks.ctrl);
        assert!(!ks.alt);
        assert!(!ks.shift);
        assert!(!ks.meta);
    }

    #[test]
    fn test_parse_keystroke_ctrl_c() {
        let ks = parse_keystroke("ctrl+c").unwrap();
        assert_eq!(ks.key, "c");
        assert!(ks.ctrl);
        assert!(!ks.alt);
    }

    #[test]
    fn test_parse_keystroke_ctrl_shift_enter() {
        let ks = parse_keystroke("ctrl+shift+enter").unwrap();
        assert_eq!(ks.key, "enter");
        assert!(ks.ctrl);
        assert!(ks.shift);
        assert!(!ks.alt);
    }

    #[test]
    fn test_parse_keystroke_normalizes_esc() {
        let ks = parse_keystroke("esc").unwrap();
        assert_eq!(ks.key, "escape");
    }

    #[test]
    fn test_parse_keystroke_normalizes_return() {
        let ks = parse_keystroke("return").unwrap();
        assert_eq!(ks.key, "enter");
    }

    #[test]
    fn test_parse_keystroke_empty_returns_none() {
        assert!(parse_keystroke("ctrl+").is_none());
        assert!(parse_keystroke("").is_none());
    }

    #[test]
    fn test_parse_chord_single() {
        let chord = parse_chord("ctrl+c").unwrap();
        assert_eq!(chord.len(), 1);
        assert_eq!(chord[0].key, "c");
        assert!(chord[0].ctrl);
    }

    #[test]
    fn test_parse_chord_multi() {
        let chord = parse_chord("ctrl+k ctrl+d").unwrap();
        assert_eq!(chord.len(), 2);
        assert_eq!(chord[0].key, "k");
        assert_eq!(chord[1].key, "d");
        assert!(chord[0].ctrl);
        assert!(chord[1].ctrl);
    }

    #[test]
    fn test_parse_chord_empty_returns_none() {
        assert!(parse_chord("").is_none());
    }

    #[test]
    fn test_default_bindings_not_empty() {
        let bindings = default_bindings();
        assert!(!bindings.is_empty());
    }

    #[test]
    fn test_default_bindings_contains_ctrl_c() {
        let bindings = default_bindings();
        let ctrl_c = bindings.iter().find(|b| {
            b.chord.len() == 1
                && b.chord[0].ctrl
                && b.chord[0].key == "c"
                && b.context == KeyContext::Global
        });
        assert!(ctrl_c.is_some());
        assert_eq!(ctrl_c.unwrap().action.as_deref(), Some("session.abort"));
    }

    #[test]
    fn test_resolver_simple_action() {
        let user = UserKeybindings::default();
        let mut resolver = KeybindingResolver::new(&user);
        let ks = parse_keystroke("ctrl+c").unwrap();
        let result = resolver.process(ks, &KeyContext::Global);
        assert!(matches!(result, KeybindingResult::Action(ref a) if a == "session.abort"));
    }

    #[test]
    fn test_resolver_no_match() {
        let user = UserKeybindings::default();
        let mut resolver = KeybindingResolver::new(&user);
        // ctrl+z has no default binding
        let ks = parse_keystroke("ctrl+z").unwrap();
        let result = resolver.process(ks, &KeyContext::Chat);
        assert!(matches!(result, KeybindingResult::NoMatch));
    }

    #[test]
    fn test_resolver_context_match_global_from_chat() {
        let user = UserKeybindings::default();
        let mut resolver = KeybindingResolver::new(&user);
        // ctrl+l in Chat context maps to input.clear
        // Global context is checked after context-specific bindings
        let ks = parse_keystroke("ctrl+l").unwrap();
        let result = resolver.process(ks, &KeyContext::Chat);
        assert!(matches!(result, KeybindingResult::Action(ref a) if a == "input.clear"));
    }

    #[test]
    fn test_keystroke_equality() {
        let ks1 = parse_keystroke("ctrl+enter").unwrap();
        let ks2 = parse_keystroke("ctrl+enter").unwrap();
        let ks3 = parse_keystroke("shift+enter").unwrap();
        assert_eq!(ks1, ks2);
        assert_ne!(ks1, ks3);
    }

    #[test]
    fn test_user_keybindings_default_empty() {
        let user = UserKeybindings::default();
        assert!(user.bindings.is_empty());
    }

    #[test]
    fn test_user_keybindings_supports_ts_block_format() {
        let user = UserKeybindings::from_json_str(
            r#"{
  "bindings": [
    {
      "context": "Chat",
      "bindings": {
        "ctrl+g": "chat:externalEditor",
        "space": null
      }
    }
  ]
}"#,
        );

        assert_eq!(user.bindings.len(), 2);
        assert_eq!(user.bindings[0].context.as_deref(), Some("Chat"));
        assert_eq!(user.bindings[0].chord, "ctrl+g");
        assert_eq!(
            user.bindings[0].action.as_deref(),
            Some("chat:externalEditor")
        );
        assert_eq!(user.bindings[1].chord, "space");
        assert_eq!(user.bindings[1].action, None);
    }

    #[test]
    fn test_new_phase1_keybindings_registered() {
        // Verify that all Phase 1 keybindings are registered
        let bindings = default_bindings();

        // Build list of keybinding actions
        let actions: Vec<String> = bindings.iter().filter_map(|b| b.action.clone()).collect();

        // Check Phase 1 keybinding actions exist
        assert!(
            actions.contains(&"input.clear".to_string()),
            "input.clear action not found"
        );
        assert!(
            actions.contains(&"chat.submit.alt".to_string()),
            "chat.submit.alt action not found"
        );
        assert!(
            actions.contains(&"transcript.issue.next".to_string()),
            "transcript.issue.next action not found"
        );
        assert!(
            actions.contains(&"transcript.issue.prev".to_string()),
            "transcript.issue.prev action not found"
        );
        assert!(
            actions.contains(&"transcript.jump.prev".to_string()),
            "transcript.jump.prev action not found"
        );
        assert!(
            actions.contains(&"transcript.jump.next".to_string()),
            "transcript.jump.next action not found"
        );
        assert!(
            actions.contains(&"openHelp".to_string())
                || actions.contains(&"help.toggle".to_string()),
            "help toggle action not found"
        );
        assert!(
            actions.contains(&"deleteCharBefore".to_string())
                || actions.contains(&"input.backspace".to_string()),
            "backspace action not found"
        );
        assert!(
            actions.contains(&"mode.cycle".to_string()),
            "mode.cycle action not found"
        );

        // Verify we have at least 10 new keybindings (Phase 1 requirement)
        assert!(
            actions.len() >= 40,
            "Expected at least 40 keybindings, found {}",
            actions.len()
        );
    }

    #[test]
    fn test_critical_parity_bindings_exist() {
        let bindings = default_bindings();
        let required: &[(&str, &str, KeyContext)] = &[
            ("enter", "chat.submit", KeyContext::Chat),
            ("shift+enter", "newline", KeyContext::Chat),
            ("ctrl+f", "search.transcript.open", KeyContext::Chat),
            ("f3", "search.transcript.next", KeyContext::Chat),
            ("shift+f3", "search.transcript.prev", KeyContext::Chat),
            ("ctrl+r", "history.overlay.open", KeyContext::Global),
            ("ctrl+c", "session.abort", KeyContext::Global),
            ("ctrl+d", "session.close", KeyContext::Global),
        ];

        for (chord_str, action, ctx) in required {
            let chord = parse_chord(chord_str).expect("required chord should parse");
            let found = bindings.iter().any(|b| {
                b.context == *ctx && b.action.as_deref() == Some(*action) && b.chord == chord
            });
            assert!(
                found,
                "Missing critical binding: {} -> {} in {:?}",
                chord_str, action, ctx
            );
        }
    }

    #[test]
    fn test_ide_bindings_add_fallback_chords() {
        let ide = ide_bindings();
        // Check that Ctrl+Q chords are present
        let has_ctrl_q_r = ide.iter().any(|b| {
            b.chord.len() == 2 && b.chord[0].key == "q" && b.chord[0].ctrl && b.chord[1].key == "r"
        });
        assert!(has_ctrl_q_r, "Ctrl+Q R should be in IDE bindings");
    }

    #[test]
    fn test_bindings_for_profile_standard_matches_default() {
        let standard = bindings_for_profile(KeybindingProfile::Standard);
        let default = default_bindings();
        assert_eq!(standard.len(), default.len());
    }

    #[test]
    fn test_bindings_for_profile_ide_includes_fallbacks() {
        let ide = bindings_for_profile(KeybindingProfile::IdeCompatible);
        let standard = bindings_for_profile(KeybindingProfile::Standard);
        assert!(ide.len() > standard.len());
    }

    #[test]
    fn test_resolver_with_ide_profile_includes_fallbacks() {
        let user = UserKeybindings::default();
        let mut resolver =
            KeybindingResolver::with_profile(&user, KeybindingProfile::IdeCompatible);

        // Test that Ctrl+Q R works in IDE profile
        let ks = parse_keystroke("ctrl+q").unwrap();
        let result = resolver.process(ks, &KeyContext::Global);
        assert!(matches!(result, KeybindingResult::Pending));

        let ks2 = parse_keystroke("r").unwrap();
        let result2 = resolver.process(ks2, &KeyContext::Global);
        assert!(matches!(result2, KeybindingResult::Action(ref a) if a == "history.overlay.open"));
    }

    #[test]
    fn test_critical_bindings_have_no_context_conflicts() {
        let bindings = default_bindings();
        let critical: &[(&str, KeyContext)] = &[
            ("enter", KeyContext::Chat),
            ("shift+enter", KeyContext::Chat),
            ("ctrl+f", KeyContext::Chat),
            ("f3", KeyContext::Chat),
            ("shift+f3", KeyContext::Chat),
            ("ctrl+r", KeyContext::Global),
        ];

        for (chord_str, ctx) in critical {
            let chord = parse_chord(chord_str).expect("critical chord should parse");
            let matches: Vec<&ParsedBinding> = bindings
                .iter()
                .filter(|b| b.context == *ctx && b.chord == chord)
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "Conflict: {} in {:?} has {} mappings",
                chord_str,
                ctx,
                matches.len()
            );
        }
    }
}
