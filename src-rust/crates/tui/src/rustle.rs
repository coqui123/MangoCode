//! Rustle mascot rendering for ratatui.
//!
//! A 5-row Unicode block-art crab-like creature. Call `rustle_lines()` to get
//! 5 `Line` values (4 body rows + 1 blank spacing row) ready for embedding in
//! a Paragraph.
//!
//! Structure (top to bottom):
//!   Row 1 — head: narrow top (5-wide) widening downward (7-wide)
//!   Row 2 — claws + eyes: widest row, pincers extend from sides
//!   Row 3 — body
//!   Row 4 — legs: body tapers into two pairs of legs via ▀ gap
//!   Row 5 — blank spacing

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// The pose / expression of the Rustle mascot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustlePose {
    Default,
    ArmsUp,
    LookLeft,
    LookRight,
}

/// Body-part style: bold pink foreground (#e91e63).
fn body_style() -> Style {
    Style::default()
        .fg(Color::Rgb(233, 30, 99))
        .add_modifier(Modifier::BOLD)
}

/// Eye-row style: pink text on black background.
fn eye_bg_style() -> Style {
    Style::default()
        .fg(Color::Rgb(233, 30, 99))
        .bg(Color::Black)
        .add_modifier(Modifier::BOLD)
}

/// Eyeball highlight style: white on black.
fn eyeball_style() -> Style {
    Style::default()
        .fg(Color::White)
        .bg(Color::Black)
        .add_modifier(Modifier::BOLD)
}

/// Build spans for the eye section, giving ▘/▝ eyeball characters white
/// foreground and everything else pink-on-black.
fn eye_spans(s: &'static str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut buf_is_eyeball = false;

    for ch in s.chars() {
        let is_eyeball = ch == '▘' || ch == '▝' || ch == '▀';
        if is_eyeball != buf_is_eyeball && !buf.is_empty() {
            let style = if buf_is_eyeball { eyeball_style() } else { eye_bg_style() };
            spans.push(Span::styled(buf.clone(), style));
            buf.clear();
        }
        buf_is_eyeball = is_eyeball;
        buf.push(ch);
    }
    if !buf.is_empty() {
        let style = if buf_is_eyeball { eyeball_style() } else { eye_bg_style() };
        spans.push(Span::styled(buf, style));
    }
    spans
}

/// Returns 5 Lines representing the Rustle mascot:
///   [0] — head row (5-wide top, 7-wide bottom)
///   [1] — claws + eyes row (widest — pincers extend from sides)
///   [2] — body row
///   [3] — legs row (body tapers into two pairs of legs)
///   [4] — blank spacing line
pub fn rustle_lines(pose: &RustlePose) -> [Line<'static>; 5] {
    // Pose varies the claw row (Row 2):
    //   r2l — left claw + head edge (body_style)
    //   r2e — eye section with ▘/▝ eyeball highlights
    //   r2r — head edge + right claw (body_style)

    let (r2l, r2e, r2r) = match pose {
        RustlePose::Default => (
            "█▄█",       // left claw tip, ▄ gap-to-connect, head edge
            "▀ █ ▘",    // keep the left eye as-is; shift only the right eye highlight left
            "█▄█",       // head edge, ▄ connect-to-gap, right claw tip
        ),
        RustlePose::ArmsUp => (
            "█▀█",       // ▀ = claw raised (upper half = arm up)
            "▀ █ ▘",    // keep the left eye as-is; shift only the right eye highlight left
            "█▀█",       // raised right claw
        ),
        RustlePose::LookLeft => (
            "█▄█",
            "▘ █ ▘",    // single-pixel upper-left quarter blocks = eyes shifted left
            "█▄█",
        ),
        RustlePose::LookRight => (
            "█▄█",
            "▝ █ ▝",    // single-pixel upper-right quarter blocks = eyes shifted right
            "█▄█",
        ),
    };

    // Row 1: head — narrow top (5-wide), wider bottom (7-wide)
    let row1 = Line::from(vec![
        Span::styled("  ▄█████▄  ".to_string(), body_style()),
    ]);

    // Row 2: claws extending from sides + face with eyeball highlights (widest row)
    let mut row2_spans = vec![Span::styled(r2l.to_string(), body_style())];
    row2_spans.extend(eye_spans(r2e));
    row2_spans.push(Span::styled(r2r.to_string(), body_style()));
    let row2 = Line::from(row2_spans);

    // Row 3: body
    let row3 = Line::from(vec![
        Span::styled(" ████████  ".to_string(), body_style()),
    ]);

    // Row 4: legs — upper half body (6-wide), lower half two leg pairs (2+gap+2)
    let row4 = Line::from(vec![
        Span::styled("  ██▀▀██   ".to_string(), body_style()),
    ]);

    // Row 5: blank spacing
    let row5 = Line::from("");

    [row1, row2, row3, row4, row5]
}
