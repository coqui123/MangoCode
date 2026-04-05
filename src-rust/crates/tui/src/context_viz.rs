// context_viz.rs — Context window and rate-limit visualization overlay.
// Triggered by the /context command. Shows horizontal progress bars.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap};
use ratatui::Frame;

use crate::overlays::centered_rect;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct ContextVizState {
    pub visible: bool,
}

impl ContextVizState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self) {
        self.visible = true;
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn render_context_viz(
    frame: &mut Frame,
    state: &ContextVizState,
    area: Rect,
    context_used: u64,
    context_total: u64,
    rate_5h: Option<f32>,
    rate_7d: Option<f32>,
    cost_usd: f64,
) {
    if !state.visible {
        return;
    }

    let dialog_width = 68u16.min(area.width.saturating_sub(4));
    let dialog_height = 20u16.min(area.height.saturating_sub(4));
    let dialog_area = centered_rect(dialog_width, dialog_height, area);

    frame.render_widget(Clear, dialog_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Line::from(vec![Span::styled(
            " Context & Usage ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(dialog_area);
    frame.render_widget(block, dialog_area);

    // bar_width: leave room for "  label  [" prefix (14 chars) and "] 100%" suffix (6 chars)
    let bar_width = (inner.width as usize).saturating_sub(22).max(4);

    let ctx_pct = if context_total > 0 {
        (context_used as f32 / context_total as f32).min(1.0)
    } else {
        0.0
    };
    let ctx_color = if ctx_pct > 0.95 {
        Color::Red
    } else if ctx_pct > 0.80 {
        Color::Yellow
    } else {
        Color::Green
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(""));

    // -- Context window ----------------------------------------------------------
    lines.push(Line::from(vec![Span::styled(
        "  Context Window",
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    )]));

    let filled = ((ctx_pct * bar_width as f32) as usize).min(bar_width);
    let empty = bar_width - filled;
    lines.push(Line::from(vec![
        Span::styled("  [", Style::default().fg(Color::DarkGray)),
        Span::styled("\u{2588}".repeat(filled), Style::default().fg(ctx_color)),
        Span::styled("\u{2591}".repeat(empty), Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("]  {:.0}%  ({} / {})",
                ctx_pct * 100.0,
                format_tokens(context_used),
                format_tokens(context_total),
            ),
            Style::default().fg(ctx_color),
        ),
    ]));

    lines.push(Line::from(""));

    // -- Rate limits -------------------------------------------------------------
    lines.push(Line::from(vec![Span::styled(
        "  Rate Limits",
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    )]));

    for (label, pct_opt) in &[("  5-hour ", rate_5h), ("  7-day  ", rate_7d)] {
        match pct_opt {
            Some(pct) => {
                let p = pct.clamp(0.0, 1.0);
                let color = if p > 0.90 {
                    Color::Red
                } else if p > 0.70 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let f = ((p * bar_width as f32) as usize).min(bar_width);
                let e = bar_width - f;
                lines.push(Line::from(vec![
                    Span::styled(label.to_string(), Style::default().fg(Color::White)),
                    Span::styled("  [", Style::default().fg(Color::DarkGray)),
                    Span::styled("\u{2588}".repeat(f), Style::default().fg(color)),
                    Span::styled("\u{2591}".repeat(e), Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("]  {:.0}%", p * 100.0),
                        Style::default().fg(color),
                    ),
                ]));
            }
            None => {
                lines.push(Line::from(vec![
                    Span::styled(label.to_string(), Style::default().fg(Color::White)),
                    Span::styled("  no data", Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
    }

    lines.push(Line::from(""));

    // -- Cost --------------------------------------------------------------------
    lines.push(Line::from(vec![
        Span::styled("  Session cost:  ", Style::default().fg(Color::White)),
        Span::styled(
            format!("${:.4}", cost_usd),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
    ]));

    lines.push(Line::from(""));
    lines.push(Line::from(vec![Span::styled(
        "  Press Esc or Enter to close",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    )]));

    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(inner, frame.buffer_mut());
}

fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
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
    fn context_viz_defaults_hidden() {
        let state = ContextVizState::new();
        assert!(!state.visible);
    }

    #[test]
    fn context_viz_toggle() {
        let mut state = ContextVizState::new();
        state.toggle();
        assert!(state.visible);
        state.toggle();
        assert!(!state.visible);
    }

    #[test]
    fn context_viz_renders_without_panic() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut state = ContextVizState::new();
        state.open();
        terminal.draw(|frame| {
            render_context_viz(frame, &state, frame.area(), 50_000, 200_000, Some(0.3), Some(0.1), 0.42);
        }).unwrap();
        let content: String = terminal.backend().buffer().clone().content().iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("Context") || content.contains("Rate"));
    }

    #[test]
    fn context_viz_hidden_renders_nothing() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let state = ContextVizState::new();
        let before = terminal.backend().buffer().clone();
        terminal.draw(|frame| {
            render_context_viz(frame, &state, frame.area(), 0, 0, None, None, 0.0);
        }).unwrap();
        assert_eq!(terminal.backend().buffer().content(), before.content());
    }
}
