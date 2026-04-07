// export_dialog.rs — Format picker dialog for /export command.
//
// Shows a two-option dialog (JSON | Markdown). On confirm, caller writes the file.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap};
use ratatui::Frame;

use crate::overlays::centered_rect;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    #[default]
    Json,
    Markdown,
}

#[derive(Debug, Default, Clone)]
pub struct ExportDialogState {
    pub visible: bool,
    pub selected: ExportFormat,
}

impl ExportDialogState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self) {
        self.visible = true;
        self.selected = ExportFormat::default();
    }

    pub fn dismiss(&mut self) {
        self.visible = false;
    }

    pub fn toggle(&mut self) {
        self.selected = match self.selected {
            ExportFormat::Json => ExportFormat::Markdown,
            ExportFormat::Markdown => ExportFormat::Json,
        };
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn render_export_dialog(frame: &mut Frame, state: &ExportDialogState, area: Rect) {
    if !state.visible {
        return;
    }

    let dialog_width = 58u16.min(area.width.saturating_sub(4));
    let dialog_height = 12u16.min(area.height.saturating_sub(4));
    let dialog_area = centered_rect(dialog_width, dialog_height, area);

    frame.render_widget(Clear, dialog_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Line::from(vec![Span::styled(
            " Export Conversation ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(dialog_area);
    frame.render_widget(block, dialog_area);

    let json_style = if state.selected == ExportFormat::Json {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    } else {
        Style::default().fg(Color::White)
    };
    let md_style = if state.selected == ExportFormat::Markdown {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    } else {
        Style::default().fg(Color::White)
    };

    let lines: Vec<Line<'static>> = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Choose export format:",
            Style::default().fg(Color::Yellow),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  [1] ", Style::default().fg(Color::DarkGray)),
            Span::styled("JSON        ", json_style),
            Span::styled("  [2] ", Style::default().fg(Color::DarkGray)),
            Span::styled("Markdown", md_style),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Saved to: ./claude-export-<timestamp>.<ext>",
            Style::default().fg(Color::DarkGray),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Tab/\u{2190}\u{2192} to switch  \u{b7}  Enter to export  \u{b7}  Esc to cancel",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )]),
    ];

    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(inner, frame.buffer_mut());
}

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

pub fn export_as_markdown(
    messages: &[mangocode_core::types::Message],
    session_title: Option<&str>,
) -> String {
    use mangocode_core::types::Role;
    let mut out = String::new();
    if let Some(title) = session_title {
        out.push_str(&format!("# {}\n\n", title));
    } else {
        out.push_str("# MangoCode Conversation Export\n\n");
    }
    for msg in messages {
        let label = match msg.role {
            Role::User => "**User**",
            Role::Assistant => "**MangoCode**",
        };
        let text = msg.get_all_text();
        out.push_str(&format!("{}\n\n{}\n\n---\n\n", label, text));
    }
    out
}

pub fn export_as_json(
    messages: &[mangocode_core::types::Message],
    session_title: Option<&str>,
) -> serde_json::Value {
    use mangocode_core::types::Role;
    let items: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": match m.role { Role::User => "user", Role::Assistant => "assistant" },
                "content": m.get_all_text(),
            })
        })
        .collect();
    serde_json::json!({
        "title": session_title,
        "messages": items,
        "exported_at": chrono::Local::now().to_rfc3339(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn export_dialog_defaults_hidden() {
        let state = ExportDialogState::new();
        assert!(!state.visible);
        assert_eq!(state.selected, ExportFormat::Json);
    }

    #[test]
    fn export_dialog_open() {
        let mut state = ExportDialogState::new();
        state.open();
        assert!(state.visible);
    }

    #[test]
    fn export_dialog_toggle() {
        let mut state = ExportDialogState::new();
        state.open();
        assert_eq!(state.selected, ExportFormat::Json);
        state.toggle();
        assert_eq!(state.selected, ExportFormat::Markdown);
        state.toggle();
        assert_eq!(state.selected, ExportFormat::Json);
    }

    #[test]
    fn export_dialog_renders_without_panic() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut state = ExportDialogState::new();
        state.open();
        terminal
            .draw(|frame| {
                render_export_dialog(frame, &state, frame.area());
            })
            .unwrap();
        let content: String = terminal
            .backend()
            .buffer()
            .clone()
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Export") || content.contains("JSON"));
    }

    #[test]
    fn export_dialog_hidden_renders_nothing() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let state = ExportDialogState::new();
        let before = terminal.backend().buffer().clone();
        terminal
            .draw(|frame| {
                render_export_dialog(frame, &state, frame.area());
            })
            .unwrap();
        assert_eq!(terminal.backend().buffer().content(), before.content());
    }
}
