//! Markdown -> ratatui lines renderer used by transcript message families.

use crate::figures;
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;


/// Render markdown text to styled ratatui lines.
pub fn render_markdown(text: &str, width: u16) -> Vec<Line<'static>> {
    let all_lines: Vec<&str> = text.lines().collect();
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut idx = 0;

    while idx < all_lines.len() {
        let raw = all_lines[idx];
        if raw.trim_start().starts_with("```") {
            if in_code_block {
                lines.push(Line::from(vec![Span::styled(
                    "  └──────────────────────────────────────────────────".to_string(),
                    Style::default().fg(Color::Yellow),
                )]));
                in_code_block = false;
                code_lang.clear();
            } else {
                in_code_block = true;
                code_lang = raw.trim_start().trim_start_matches('`').trim().to_string();
                let lang_label = if code_lang.is_empty() {
                    String::new()
                } else {
                    format!(" {} ", code_lang)
                };
                lines.push(Line::from(vec![Span::styled(
                    format!("  ┌──────────────────────{}", lang_label),
                    Style::default().fg(Color::Yellow),
                )]));
            }
            idx += 1;
            continue;
        }

        if in_code_block {
            lines.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::Yellow)),
                Span::styled(raw.to_string(), Style::default().fg(Color::White)),
            ]));
            idx += 1;
            continue;
        }

        // Check for markdown tables
        if let Some((table, end_idx)) = super::markdown_enhanced::detect_table(&all_lines, idx) {
            lines.extend(super::markdown_enhanced::render_table(&table));
            idx = end_idx;
            continue;
        }

        if let Some(quoted) = raw.strip_prefix("> ") {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", figures::BLOCKQUOTE_BAR),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(quoted.to_string(), Style::default().fg(Color::DarkGray)),
            ]));
            idx += 1;
            continue;
        }

        if let Some(stripped) = raw.strip_prefix("### ") {
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", stripped),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )]));
            idx += 1;
            continue;
        }
        if let Some(stripped) = raw.strip_prefix("## ") {
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", stripped),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )]));
            idx += 1;
            continue;
        }
        if let Some(stripped) = raw.strip_prefix("# ") {
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", stripped),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD | Modifier::ITALIC | Modifier::UNDERLINED),
            )]));
            idx += 1;
            continue;
        }

        let padded = format!("  {}", raw);
        let effective_width = width.saturating_sub(4) as usize;
        for wrapped_line in word_wrap(&padded, effective_width) {
            let spans = super::markdown_enhanced::parse_inline_formatting(&wrapped_line);
            lines.push(Line::from(spans));
        }

        idx += 1;
    }

    if in_code_block {
        lines.push(Line::from(vec![Span::styled(
            "  └──────────────────────────────────────────────────".to_string(),
            Style::default().fg(Color::Yellow),
        )]));
    }

    lines
}


fn word_wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 || UnicodeWidthStr::width(text) <= width {
        return vec![text.to_string()];
    }

    // Preserve leading whitespace so indented content keeps its indent.
    let leading: &str = &text[..text.len() - text.trim_start().len()];
    let leading_w = UnicodeWidthStr::width(leading);
    let body = &text[leading.len()..];

    let mut result = Vec::new();
    let mut current_line = String::from(leading);
    let mut current_width = leading_w;
    let mut first = true;

    for word in body.split_whitespace() {
        let word_w = UnicodeWidthStr::width(word);
        if first {
            current_line.push_str(word);
            current_width += word_w;
            first = false;
        } else if current_width + 1 + word_w <= width {
            current_line.push(' ');
            current_line.push_str(word);
            current_width += 1 + word_w;
        } else {
            result.push(std::mem::take(&mut current_line));
            // Continuation lines get same leading indent
            current_line.push_str(leading);
            current_line.push_str(word);
            current_width = leading_w + word_w;
        }
    }

    if !current_line.is_empty() {
        result.push(current_line);
    }
    if result.is_empty() {
        result.push(text.to_string());
    }
    result
}
