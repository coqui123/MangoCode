//! Main-buffer hybrid TUI.
//!
//! The transcript is real terminal output, so terminal-native scrollback works.
//! The live composer is rendered from ratatui widgets into an offscreen buffer
//! and repainted in reserved bottom rows.

use std::io::{self, Stdout, Write};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute, queue,
    style::{
        Attribute as CtAttribute, Color as CtColor, Print, ResetColor, SetAttribute,
        SetBackgroundColor, SetForegroundColor,
    },
    terminal::{
        self, disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use mangocode_api::{streaming::ContentDelta, AnthropicStreamEvent};
use mangocode_core::types::{ContentBlock, Message, MessageContent};
use mangocode_query::QueryEvent;
use ratatui::{
    backend::TestBackend,
    buffer::Buffer,
    layout::Rect,
    style::{Color as RtColor, Modifier, Style as RtStyle},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
    Terminal,
};
use unicode_width::UnicodeWidthStr;

use crate::{
    app::App,
    messages::{render_message, RenderContext},
    prompt_input::{input_height, render_prompt_input, InputMode},
    render::{build_tool_names, flush_sixel_blit_with_cursor, render_app, reset_sixel_blit_state},
};

const MANGO: RtColor = RtColor::Rgb(255, 176, 32);
const DARK: RtColor = RtColor::Rgb(26, 20, 15);
const TEXT: RtColor = RtColor::Rgb(253, 246, 227);
const MUTED: RtColor = RtColor::Rgb(138, 125, 115);

pub struct HybridTerminal {
    stdout: Stdout,
    composer_height: u16,
    transcript_row: u16,
    stream_open: bool,
    printed_messages: usize,
    overlay_active: bool,
    composer_buffer: Option<Buffer>,
    fullscreen_buffer: Option<Buffer>,
    screen_lines: HashMap<u16, (Line<'static>, Option<String>)>,
    expansion_fingerprint: u64,
    deferred_new_messages: bool,
    restored: bool,
}

impl HybridTerminal {
    pub fn setup(app: &App) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // Keep mouse reporting off in main-buffer transcript mode so the
        // terminal owns wheel scrolling and native scrollback keeps working.
        execute!(stdout, EnableBracketedPaste, Hide)?;
        queue_base_style(&mut stdout)?;
        execute!(stdout, Clear(ClearType::All), MoveTo(0, 0))?;
        reset_sixel_blit_state();
        let mut terminal = Self {
            stdout,
            composer_height: 0,
            transcript_row: 0,
            stream_open: false,
            printed_messages: 0,
            overlay_active: false,
            composer_buffer: None,
            fullscreen_buffer: None,
            screen_lines: HashMap::new(),
            expansion_fingerprint: expansion_fingerprint(app),
            deferred_new_messages: false,
            restored: false,
        };
        terminal.render_live(app)?;
        Ok(terminal)
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.leave_fullscreen(false)?;
        self.reset_scroll_region()?;
        self.clear_composer_area()?;
        execute!(
            self.stdout,
            Show,
            DisableMouseCapture,
            DisableBracketedPaste,
            ResetColor
        )?;
        disable_raw_mode()?;
        self.restored = true;
        Ok(())
    }

    pub fn suspend(&mut self) -> io::Result<()> {
        self.restore()
    }

    pub fn resume(app: &App, messages: &[Message]) -> io::Result<Self> {
        let mut terminal = Self::setup(app)?;
        terminal.print_initial_messages(app, messages)?;
        Ok(terminal)
    }

    pub fn reset_transcript(&mut self, app: &App, messages: &[Message]) -> io::Result<()> {
        if app.active_modal_owner().is_some() || messages.is_empty() {
            self.printed_messages = 0;
            self.transcript_row = 0;
            self.stream_open = false;
            self.composer_buffer = None;
            self.fullscreen_buffer = None;
            self.screen_lines.clear();
            self.deferred_new_messages = false;
            app.tool_row_map.borrow_mut().clear();
            app.message_row_map.borrow_mut().clear();
            return self.render_live(app);
        }

        self.leave_fullscreen(true)?;
        self.reset_scroll_region()?;
        queue_base_style(&mut self.stdout)?;
        execute!(self.stdout, Clear(ClearType::All), MoveTo(0, 0))?;
        reset_sixel_blit_state();
        self.composer_height = 0;
        self.transcript_row = 0;
        self.stream_open = false;
        self.printed_messages = 0;
        self.composer_buffer = None;
        self.fullscreen_buffer = None;
        self.screen_lines.clear();
        self.deferred_new_messages = false;
        app.tool_row_map.borrow_mut().clear();
        app.message_row_map.borrow_mut().clear();
        self.print_initial_messages(app, messages)
    }

    pub fn render_live(&mut self, app: &App) -> io::Result<()> {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        if app.active_modal_owner().is_some()
            || (app.messages.is_empty() && self.printed_messages == 0)
        {
            self.render_fullscreen_app(app, cols, rows)?;
            return Ok(());
        }
        if self.overlay_active {
            self.leave_fullscreen(false)?;
        }
        if self.deferred_new_messages && !app.is_streaming {
            self.deferred_new_messages = false;
            self.print_new_messages(app)?;
        }
        let expansion_fingerprint = expansion_fingerprint(app);
        if expansion_fingerprint != self.expansion_fingerprint {
            if app.is_streaming {
                self.expansion_fingerprint = expansion_fingerprint;
            } else {
                self.reset_transcript(app, &app.messages)?;
                self.expansion_fingerprint = expansion_fingerprint;
                return Ok(());
            }
        }
        let height = composer_height(app).min(rows.saturating_sub(1)).max(3);
        app.last_msg_area
            .set(Rect::new(0, 0, cols, rows.saturating_sub(height)));
        app.last_selectable_area
            .set(Rect::new(0, 0, cols, rows.saturating_sub(height)));
        if height != self.composer_height {
            self.reset_scroll_region()?;
            self.clear_composer_area()?;
            self.composer_height = height;
            self.composer_buffer = None;
            self.apply_scroll_region()?;
        }

        let top = rows.saturating_sub(self.composer_height);
        let area = Rect::new(0, 0, cols, self.composer_height);
        let mut buffer = Buffer::empty(area);
        render_composer_to_buffer(app, area, &mut buffer);
        let previous = self.composer_buffer.clone();
        self.flush_buffer_diff(top, &buffer, previous.as_ref())?;
        self.composer_buffer = Some(buffer);
        let (row, col) = composer_cursor(app, cols, top);
        queue!(self.stdout, MoveTo(col, row))?;
        self.stdout.flush()
    }

    pub fn print_initial_messages(&mut self, app: &App, messages: &[Message]) -> io::Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        self.prepare_for_transcript(app)?;
        app.tool_row_map.borrow_mut().clear();
        app.message_row_map.borrow_mut().clear();
        self.screen_lines.clear();
        self.move_to_transcript_end()?;
        for (idx, msg) in messages.iter().enumerate() {
            app.message_row_map
                .borrow_mut()
                .insert(self.transcript_row, idx);
            self.print_message(msg, app)?;
        }
        self.printed_messages = messages.len();
        self.expansion_fingerprint = expansion_fingerprint(app);
        self.deferred_new_messages = false;
        self.render_live(app)
    }

    pub fn print_user_input(&mut self, app: &App, input: &str) -> io::Result<()> {
        self.prepare_for_transcript(app)?;
        self.move_to_transcript_end()?;
        app.message_row_map
            .borrow_mut()
            .insert(self.transcript_row, self.printed_messages);
        let msg = Message::user(input.to_string());
        self.print_message(&msg, app)?;
        self.printed_messages = self.printed_messages.saturating_add(1);
        self.render_live(app)
    }

    pub fn print_assistant_text(&mut self, app: &App, text: &str) -> io::Result<()> {
        self.prepare_for_transcript(app)?;
        self.move_to_transcript_end()?;
        app.message_row_map
            .borrow_mut()
            .insert(self.transcript_row, self.printed_messages);
        let msg = Message::assistant(text.to_string());
        self.print_message(&msg, app)?;
        self.printed_messages = self.printed_messages.saturating_add(1);
        self.render_live(app)
    }

    pub fn print_message(&mut self, msg: &Message, app: &App) -> io::Result<()> {
        let (cols, _) = terminal::size().unwrap_or((80, 24));
        let mut ctx = RenderContext::default();
        ctx.width = cols.saturating_sub(1).max(1);
        ctx.tool_names = build_tool_names(&app.messages);
        ctx.expanded_thinking = app.thinking_expanded.clone();
        ctx.expanded_tool_outputs = app.expanded_tool_outputs.clone();
        let lines = render_message(msg, &ctx);
        let tool_row_ids = tool_row_ids_for_message(msg, &lines, app);
        for (line, tool_id) in lines.into_iter().zip(tool_row_ids) {
            self.write_line_with_tool(&line, tool_id.as_deref(), Some(app))?;
        }
        self.write_blank_line_for_app(app)?;
        Ok(())
    }

    pub fn handle_query_event(&mut self, app: &App, event: &QueryEvent) -> io::Result<()> {
        if app.active_modal_owner().is_some() {
            self.deferred_new_messages = true;
            return self.render_live(app);
        }

        match event {
            QueryEvent::Stream(stream_evt)
            | QueryEvent::StreamWithParent {
                event: stream_evt, ..
            } => self.handle_stream_event(app, stream_evt),
            QueryEvent::ToolStart {
                tool_name,
                tool_id,
                input_json,
                ..
            } => {
                self.move_to_transcript_end()?;
                if self.stream_open {
                    self.write_blank_line_for_app(app)?;
                    self.stream_open = false;
                }
                self.write_styled_line_with_tool(
                    vec![
                        Span::styled("  • ", RtStyle::default().fg(MANGO)),
                        Span::styled(
                            format!("Running {}", tool_name),
                            RtStyle::default().fg(TEXT).add_modifier(Modifier::BOLD),
                        ),
                    ],
                    Some(tool_id.as_str()),
                    app,
                )?;
                if !input_json.trim().is_empty() && input_json.trim() != "null" {
                    self.write_styled_line_for_app(
                        vec![Span::styled(
                            "    input",
                            RtStyle::default().fg(MUTED).add_modifier(Modifier::DIM),
                        )],
                        app,
                    )?;
                    self.write_wrapped_styled_for_app(
                        app,
                        &truncate(input_json, 1200),
                        "    ",
                        RtStyle::default().fg(MUTED),
                    )?;
                }
                self.render_live(app)
            }
            QueryEvent::ToolEnd {
                tool_name,
                tool_id,
                result,
                is_error,
                ..
            } => {
                self.move_to_transcript_end()?;
                let color = if *is_error { RtColor::Red } else { MANGO };
                let label = if *is_error { "Tool error" } else { "Tool done" };
                self.write_styled_line_with_tool(
                    vec![
                        Span::styled("  • ", RtStyle::default().fg(color)),
                        Span::styled(
                            format!("{}: {}", label, tool_name),
                            RtStyle::default().fg(TEXT).add_modifier(Modifier::BOLD),
                        ),
                    ],
                    Some(tool_id.as_str()),
                    app,
                )?;
                let mut line_count = 0;
                for line in result.lines().take(8) {
                    line_count += 1;
                    self.write_wrapped_styled_for_app(
                        app,
                        line,
                        "    ",
                        RtStyle::default().fg(MUTED),
                    )?;
                }
                if line_count == 0 && !result.is_empty() {
                    self.write_wrapped_styled_for_app(
                        app,
                        result,
                        "    ",
                        RtStyle::default().fg(MUTED),
                    )?;
                }
                if result.lines().count() > 8 {
                    self.write_styled_line_for_app(
                        vec![Span::styled(
                            "    ...".to_string(),
                            RtStyle::default().fg(MUTED).add_modifier(Modifier::DIM),
                        )],
                        app,
                    )?;
                }
                self.render_live(app)
            }
            QueryEvent::TurnComplete { .. } => {
                self.move_to_transcript_end()?;
                if self.stream_open {
                    self.write_blank_line_for_app(app)?;
                    self.stream_open = false;
                }
                self.render_live(app)
            }
            QueryEvent::Status(_) | QueryEvent::TokenWarning { .. } => self.render_live(app),
            QueryEvent::Error(msg) => {
                self.move_to_transcript_end()?;
                if self.stream_open {
                    self.write_blank_line_for_app(app)?;
                    self.stream_open = false;
                }
                self.write_styled_line_for_app(
                    vec![Span::styled(
                        format!("Error: {}", msg),
                        RtStyle::default()
                            .fg(RtColor::Red)
                            .add_modifier(Modifier::BOLD),
                    )],
                    app,
                )?;
                self.printed_messages = app.messages.len();
                self.render_live(app)
            }
        }
    }

    fn handle_stream_event(&mut self, app: &App, event: &AnthropicStreamEvent) -> io::Result<()> {
        match event {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => {
                // Keep raw provider deltas out of scrollback. The committed
                // message renderer needs the completed text to render markdown
                // cleanly, so we print the assistant turn on MessageStop/
                // TurnComplete via `print_new_messages`.
                let _ = text;
                self.render_live(app)
            }
            AnthropicStreamEvent::MessageStop => {
                self.move_to_transcript_end()?;
                self.stream_open = false;
                self.render_live(app)
            }
            _ => Ok(()),
        }
    }

    fn prepare_for_transcript(&mut self, app: &App) -> io::Result<()> {
        if self.overlay_active && app.active_modal_owner().is_none() {
            self.leave_fullscreen(true)?;
            self.reset_scroll_region()?;
            queue_base_style(&mut self.stdout)?;
            execute!(self.stdout, Clear(ClearType::All), MoveTo(0, 0))?;
            reset_sixel_blit_state();
            self.composer_height = 0;
            self.transcript_row = 0;
            self.stream_open = false;
            self.composer_buffer = None;
            self.fullscreen_buffer = None;
            self.screen_lines.clear();
        }
        Ok(())
    }

    pub fn print_new_messages(&mut self, app: &App) -> io::Result<()> {
        if self.printed_messages >= app.messages.len() {
            return Ok(());
        }
        if app.active_modal_owner().is_some() {
            self.deferred_new_messages = true;
            return self.render_live(app);
        }
        self.prepare_for_transcript(app)?;
        self.move_to_transcript_end()?;
        let start = self.printed_messages;
        for (offset, msg) in app.messages[start..].iter().enumerate() {
            if let Some(msg) = live_transcript_message(msg) {
                app.message_row_map
                    .borrow_mut()
                    .insert(self.transcript_row, start + offset);
                self.print_message(&msg, app)?;
            }
        }
        self.printed_messages = app.messages.len();
        Ok(())
    }

    fn apply_scroll_region(&mut self) -> io::Result<()> {
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let bottom = rows.saturating_sub(self.composer_height).max(1);
        write!(self.stdout, "\x1b[1;{}r", bottom)?;
        self.stdout.flush()
    }

    fn reset_scroll_region(&mut self) -> io::Result<()> {
        write!(self.stdout, "\x1b[r")?;
        self.stdout.flush()
    }

    fn move_to_transcript_end(&mut self) -> io::Result<()> {
        self.apply_scroll_region()?;
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let bottom = rows.saturating_sub(self.composer_height).saturating_sub(1);
        self.transcript_row = self.transcript_row.min(bottom);
        queue!(self.stdout, MoveTo(0, self.transcript_row))?;
        self.stdout.flush()
    }

    fn clear_composer_area(&mut self) -> io::Result<()> {
        if self.composer_height == 0 {
            return Ok(());
        }
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let top = rows.saturating_sub(self.composer_height);
        for i in 0..self.composer_height {
            queue_base_style(&mut self.stdout)?;
            queue!(
                self.stdout,
                MoveTo(0, top.saturating_add(i)),
                Clear(ClearType::CurrentLine)
            )?;
        }
        self.stdout.flush()
    }

    fn flush_buffer(&mut self, screen_y: u16, buffer: &Buffer) -> io::Result<()> {
        for y in 0..buffer.area.height {
            queue!(self.stdout, MoveTo(0, screen_y.saturating_add(y)))?;
            for x in 0..buffer.area.width {
                let cell = &buffer[(x, y)];
                queue_style(&mut self.stdout, cell.style())?;
                queue!(self.stdout, Print(cell.symbol()))?;
            }
            queue!(self.stdout, ResetColor, SetAttribute(CtAttribute::Reset))?;
        }
        Ok(())
    }

    fn flush_buffer_diff(
        &mut self,
        screen_y: u16,
        buffer: &Buffer,
        previous: Option<&Buffer>,
    ) -> io::Result<()> {
        let Some(previous) = previous else {
            return self.flush_buffer(screen_y, buffer);
        };
        if previous.area != buffer.area {
            return self.flush_buffer(screen_y, buffer);
        }

        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                let next = &buffer[(x, y)];
                let prev = &previous[(x, y)];
                if next.symbol() == prev.symbol() && next.style() == prev.style() {
                    continue;
                }
                queue!(self.stdout, MoveTo(x, screen_y.saturating_add(y)))?;
                queue_style(&mut self.stdout, next.style())?;
                queue!(self.stdout, Print(next.symbol()))?;
            }
        }
        queue!(self.stdout, ResetColor, SetAttribute(CtAttribute::Reset))?;
        Ok(())
    }

    fn write_line_with_tool(
        &mut self,
        line: &Line<'_>,
        tool_output_id: Option<&str>,
        app: Option<&App>,
    ) -> io::Result<()> {
        self.move_to_transcript_end()?;
        let row = self.transcript_row;
        self.composer_buffer = None;
        queue_base_style(&mut self.stdout)?;
        queue!(self.stdout, Clear(ClearType::CurrentLine))?;
        if let (Some(app), Some(id)) = (app, tool_output_id) {
            app.tool_row_map.borrow_mut().insert(row, id.to_string());
        }
        self.screen_lines.insert(
            row,
            (clone_line_static(line), tool_output_id.map(str::to_string)),
        );
        for span in &line.spans {
            queue_transcript_style(&mut self.stdout, span.style)?;
            let text = span.content.as_ref();
            write!(self.stdout, "{}", text)?;
        }
        queue!(self.stdout, ResetColor, SetAttribute(CtAttribute::Reset))?;
        writeln!(self.stdout)?;
        let (_, rows) = terminal::size().unwrap_or((80, 24));
        let bottom = rows.saturating_sub(self.composer_height).saturating_sub(1);
        if row >= bottom {
            self.shift_visible_rows_after_scroll(app, bottom);
            self.transcript_row = bottom;
        } else {
            self.transcript_row = row.saturating_add(1).min(bottom);
        }
        self.stdout.flush()
    }

    fn write_styled_line_for_app(
        &mut self,
        spans: Vec<Span<'static>>,
        app: &App,
    ) -> io::Result<()> {
        self.write_line_with_tool(&Line::from(spans), None, Some(app))
    }

    fn write_styled_line_with_tool(
        &mut self,
        spans: Vec<Span<'static>>,
        tool_output_id: Option<&str>,
        app: &App,
    ) -> io::Result<()> {
        self.write_line_with_tool(&Line::from(spans), tool_output_id, Some(app))
    }

    fn write_blank_line_for_app(&mut self, app: &App) -> io::Result<()> {
        self.write_styled_line_for_app(vec![Span::raw("")], app)
    }

    fn write_wrapped_styled_for_app(
        &mut self,
        app: &App,
        text: &str,
        indent: &str,
        style: RtStyle,
    ) -> io::Result<()> {
        let width = terminal::size().unwrap_or((80, 24)).0 as usize;
        let indent_width = UnicodeWidthStr::width(indent);
        let content_width = width.saturating_sub(indent_width).max(12);

        for logical in text.lines() {
            for wrapped in wrap_visual(logical, content_width) {
                self.write_styled_line_for_app(
                    vec![
                        Span::styled(indent.to_string(), style),
                        Span::styled(wrapped, style),
                    ],
                    app,
                )?;
            }
        }
        Ok(())
    }

    fn shift_visible_rows_after_scroll(&mut self, app: Option<&App>, bottom: u16) {
        self.screen_lines = self
            .screen_lines
            .drain()
            .filter_map(|(row, value)| {
                if row == 0 || row > bottom {
                    None
                } else {
                    Some((row - 1, value))
                }
            })
            .collect();

        if let Some(app) = app {
            shift_row_map(&mut app.message_row_map.borrow_mut(), bottom);
            shift_row_map(&mut app.tool_row_map.borrow_mut(), bottom);
        }
    }

    fn render_fullscreen_app(&mut self, app: &App, cols: u16, rows: u16) -> io::Result<()> {
        self.enter_fullscreen()?;

        let backend = TestBackend::new(cols, rows);
        let mut terminal = Terminal::new(backend)?;
        terminal.draw(|frame| render_app(frame, app))?;

        let cursor = terminal.get_cursor_position()?;
        let buffer = terminal.backend().buffer().clone();
        let previous = self.fullscreen_buffer.clone();
        self.flush_buffer_diff(0, &buffer, previous.as_ref())?;
        self.fullscreen_buffer = Some(buffer);
        queue!(self.stdout, MoveTo(cursor.x, cursor.y))?;
        self.stdout.flush()?;
        flush_sixel_blit_with_cursor(app, Some((cursor.y, cursor.x)));
        Ok(())
    }

    fn enter_fullscreen(&mut self) -> io::Result<()> {
        if self.overlay_active {
            return Ok(());
        }

        self.reset_scroll_region()?;
        execute!(
            self.stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            Clear(ClearType::All),
            MoveTo(0, 0)
        )?;
        queue_base_style(&mut self.stdout)?;
        reset_sixel_blit_state();
        self.overlay_active = true;
        self.composer_buffer = None;
        self.fullscreen_buffer = None;
        self.stdout.flush()
    }

    fn leave_fullscreen(&mut self, clear_main: bool) -> io::Result<()> {
        if !self.overlay_active {
            return Ok(());
        }

        self.reset_scroll_region()?;
        execute!(self.stdout, DisableMouseCapture, LeaveAlternateScreen)?;
        if clear_main {
            execute!(self.stdout, Clear(ClearType::All), MoveTo(0, 0))?;
        }
        queue_base_style(&mut self.stdout)?;
        reset_sixel_blit_state();
        self.overlay_active = false;
        if clear_main {
            self.composer_height = 0;
            self.transcript_row = 0;
        }
        self.composer_buffer = None;
        self.fullscreen_buffer = None;
        if clear_main {
            self.screen_lines.clear();
        }
        self.stdout.flush()
    }
}

impl Drop for HybridTerminal {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

fn render_composer_to_buffer(app: &App, area: Rect, buffer: &mut Buffer) {
    for y in 0..area.height {
        for x in 0..area.width {
            buffer[(x, y)]
                .set_char(' ')
                .set_style(RtStyle::default().bg(DARK).fg(TEXT));
        }
    }

    let status = composer_status_line(app);
    Paragraph::new(status).render(Rect::new(0, 0, area.width, 1), buffer);

    let input_area = Rect::new(0, 1, area.width, area.height.saturating_sub(2));
    render_prompt_input(
        &app.prompt_input,
        input_area,
        buffer,
        !app.is_streaming,
        if app.is_streaming {
            InputMode::Readonly
        } else if app.plan_mode {
            InputMode::Plan
        } else {
            InputMode::Default
        },
    );

    let footer = footer_line(app);
    Paragraph::new(footer).render(
        Rect::new(0, area.height.saturating_sub(1), area.width, 1),
        buffer,
    );
}

fn composer_status_line(app: &App) -> Line<'static> {
    let model_short = app
        .model_name
        .split_once('/')
        .map(|(_, model)| model)
        .unwrap_or(app.model_name.as_str());
    let agent_mode =
        app.agent_mode
            .as_deref()
            .unwrap_or(if app.plan_mode { "plan" } else { "build" });
    Line::from(vec![
        Span::styled(
            format!(" {} ", model_short),
            RtStyle::default().fg(TEXT).bg(DARK),
        ),
        Span::styled("·", RtStyle::default().fg(MUTED).bg(DARK)),
        Span::styled(
            format!(" {} ", agent_mode),
            RtStyle::default().fg(MANGO).bg(DARK),
        ),
        Span::styled("·", RtStyle::default().fg(MUTED).bg(DARK)),
        Span::styled(
            " Ctrl+A: model  Ctrl+K: commands",
            RtStyle::default().fg(MUTED).bg(DARK),
        ),
    ])
}

fn footer_line(app: &App) -> Line<'static> {
    let mut text = if app.is_streaming {
        "esc to interrupt".to_string()
    } else if app.prompt_input.text.is_empty() {
        "F1 for shortcuts".to_string()
    } else {
        String::new()
    };
    if let Some(status) = app.status_message.as_deref() {
        if !text.is_empty() {
            text.push_str("  ");
        }
        text.push_str(status);
    }
    if app.cost_usd > 0.0 {
        if !text.is_empty() {
            text.push_str("  ");
        }
        text.push_str(&format!("${:.4}", app.cost_usd));
    }
    Line::from(vec![Span::styled(
        text,
        RtStyle::default().fg(MUTED).bg(DARK),
    )])
}

fn composer_height(app: &App) -> u16 {
    input_height(&app.prompt_input)
        .saturating_add(2)
        .clamp(4, 10)
}

fn composer_cursor(app: &App, width: u16, top: u16) -> (u16, u16) {
    let input_top = top.saturating_add(1);
    let content_width = width.saturating_sub(4).max(1) as usize;
    let before = app
        .prompt_input
        .text
        .get(..app.prompt_input.cursor)
        .unwrap_or(app.prompt_input.text.as_str());
    let line_count = before.lines().count().saturating_sub(1) as u16;
    let last = before.lines().last().unwrap_or("");
    let col = UnicodeWidthStr::width(last).min(content_width) as u16;
    (input_top.saturating_add(line_count), col.saturating_add(2))
}

fn live_transcript_message(msg: &Message) -> Option<Message> {
    match &msg.content {
        MessageContent::Text(_) => Some(msg.clone()),
        MessageContent::Blocks(blocks) => {
            let visible = blocks
                .iter()
                .filter(|block| {
                    !matches!(
                        block,
                        ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }
                    )
                })
                .cloned()
                .collect::<Vec<_>>();
            if visible.is_empty() {
                None
            } else {
                Some(Message {
                    role: msg.role.clone(),
                    content: MessageContent::Blocks(visible),
                    uuid: msg.uuid.clone(),
                    cost: msg.cost.clone(),
                })
            }
        }
    }
}

fn expansion_fingerprint(app: &App) -> u64 {
    let mut ids = app.expanded_tool_outputs.iter().collect::<Vec<_>>();
    ids.sort();
    let mut hasher = DefaultHasher::new();
    for id in ids {
        id.hash(&mut hasher);
    }
    hasher.finish()
}

fn clone_line_static(line: &Line<'_>) -> Line<'static> {
    Line::from(
        line.spans
            .iter()
            .map(|span| Span::styled(span.content.to_string(), span.style))
            .collect::<Vec<_>>(),
    )
}

fn shift_row_map<T: Clone>(map: &mut HashMap<u16, T>, bottom: u16) {
    *map = map
        .drain()
        .filter_map(|(row, value)| {
            if row == 0 || row > bottom {
                None
            } else {
                Some((row - 1, value))
            }
        })
        .collect();
}

fn tool_row_ids_for_message(
    msg: &Message,
    lines: &[Line<'static>],
    app: &App,
) -> Vec<Option<String>> {
    let tool_ids = msg
        .content_blocks()
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let tool_names = msg
        .content_blocks()
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    if tool_ids.is_empty() {
        return vec![None; lines.len()];
    }

    let mut row_ids = vec![None; lines.len()];
    let mut next_tool = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let is_named_tool_header = tool_names
            .iter()
            .any(|name| !name.is_empty() && text.contains(name));
        let is_tool_header = is_named_tool_header || text.contains('●');
        let is_expand_hint = text.contains("ctrl+o") || text.contains("more lines");
        if !(is_tool_header || is_expand_hint) {
            continue;
        }
        let Some(id) = tool_ids.get(next_tool.min(tool_ids.len().saturating_sub(1))) else {
            continue;
        };
        row_ids[idx] = Some(id.clone());
        if idx > 0 && is_expand_hint {
            row_ids[idx - 1] = Some(id.clone());
        }
        if is_expand_hint && next_tool + 1 < tool_ids.len() {
            next_tool += 1;
        }
    }

    // Expanded output rows should also remain clickable/hoverable.
    for idx in 0..row_ids.len() {
        if row_ids[idx].is_some() {
            let id = row_ids[idx].clone();
            let mut cursor = idx + 1;
            while let (Some(active_id), Some(slot)) = (id.as_ref(), row_ids.get_mut(cursor)) {
                if slot.is_some() {
                    break;
                }
                if app.expanded_tool_outputs.contains(active_id) {
                    *slot = Some(active_id.clone());
                }
                cursor += 1;
            }
        }
    }
    row_ids
}

fn queue_base_style(stdout: &mut Stdout) -> io::Result<()> {
    queue!(
        stdout,
        SetAttribute(CtAttribute::Reset),
        SetForegroundColor(required_ct_color(TEXT)),
        SetBackgroundColor(required_ct_color(DARK))
    )
}

fn queue_transcript_style(stdout: &mut Stdout, style: RtStyle) -> io::Result<()> {
    queue_base_style(stdout)?;
    queue!(
        stdout,
        SetForegroundColor(
            style
                .fg
                .and_then(to_ct_color)
                .unwrap_or(required_ct_color(TEXT))
        ),
        SetBackgroundColor(
            style
                .bg
                .and_then(to_ct_color)
                .unwrap_or(required_ct_color(DARK))
        )
    )?;
    if style.add_modifier.contains(Modifier::BOLD) {
        queue!(stdout, SetAttribute(CtAttribute::Bold))?;
    }
    if style.add_modifier.contains(Modifier::DIM) {
        queue!(stdout, SetAttribute(CtAttribute::Dim))?;
    }
    if style.add_modifier.contains(Modifier::ITALIC) {
        queue!(stdout, SetAttribute(CtAttribute::Italic))?;
    }
    Ok(())
}

fn queue_style(stdout: &mut Stdout, style: RtStyle) -> io::Result<()> {
    queue_base_style(stdout)?;
    queue!(
        stdout,
        SetForegroundColor(
            style
                .fg
                .and_then(to_ct_color)
                .unwrap_or(required_ct_color(TEXT))
        ),
        SetBackgroundColor(
            style
                .bg
                .and_then(to_ct_color)
                .unwrap_or(required_ct_color(DARK))
        )
    )?;
    if style.add_modifier.contains(Modifier::BOLD) {
        queue!(stdout, SetAttribute(CtAttribute::Bold))?;
    }
    if style.add_modifier.contains(Modifier::DIM) {
        queue!(stdout, SetAttribute(CtAttribute::Dim))?;
    }
    if style.add_modifier.contains(Modifier::ITALIC) {
        queue!(stdout, SetAttribute(CtAttribute::Italic))?;
    }
    Ok(())
}

fn required_ct_color(color: RtColor) -> CtColor {
    to_ct_color(color).unwrap_or(CtColor::White)
}

fn to_ct_color(color: RtColor) -> Option<CtColor> {
    Some(match color {
        RtColor::Black => CtColor::Black,
        RtColor::Red => CtColor::Red,
        RtColor::Green => CtColor::Green,
        RtColor::Yellow => CtColor::Yellow,
        RtColor::Blue => CtColor::Blue,
        RtColor::Magenta => CtColor::Magenta,
        RtColor::Cyan => CtColor::Cyan,
        RtColor::Gray => CtColor::Grey,
        RtColor::DarkGray => CtColor::DarkGrey,
        RtColor::LightRed => CtColor::DarkRed,
        RtColor::LightGreen => CtColor::DarkGreen,
        RtColor::LightYellow => CtColor::DarkYellow,
        RtColor::LightBlue => CtColor::DarkBlue,
        RtColor::LightMagenta => CtColor::DarkMagenta,
        RtColor::LightCyan => CtColor::DarkCyan,
        RtColor::White => CtColor::White,
        RtColor::Rgb(r, g, b) => CtColor::Rgb { r, g, b },
        RtColor::Indexed(i) => CtColor::AnsiValue(i),
        RtColor::Reset => return None,
    })
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(max_chars).collect();
        s.push('…');
        s
    }
}

fn wrap_visual(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in text.chars() {
        let ch_width = UnicodeWidthStr::width(ch.to_string().as_str()).max(1);
        if current_width > 0 && current_width.saturating_add(ch_width) > width {
            lines.push(current);
            current = String::new();
            current_width = 0;
        }
        current.push(ch);
        current_width = current_width.saturating_add(ch_width);
    }

    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::types::ToolResultContent;

    #[test]
    fn live_transcript_message_keeps_final_text_without_replaying_tool_blocks() {
        let msg = Message::assistant_blocks(vec![
            ContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "Bash".to_string(),
                input: serde_json::json!({ "command": "echo hi" }),
            },
            ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: ToolResultContent::Text("hi".to_string()),
                is_error: Some(false),
            },
            ContentBlock::Text {
                text: "done".to_string(),
            },
        ]);

        let visible = live_transcript_message(&msg).expect("final text should remain visible");
        assert_eq!(visible.content_blocks().len(), 1);
        assert_eq!(visible.get_all_text(), "done");
    }

    #[test]
    fn shift_row_map_drops_scrolled_rows_and_keeps_visible_hits_aligned() {
        let mut map = HashMap::from([
            (0, "drop-top".to_string()),
            (1, "first".to_string()),
            (3, "third".to_string()),
            (4, "drop-below-bottom".to_string()),
        ]);

        shift_row_map(&mut map, 3);

        assert_eq!(map.get(&0).map(String::as_str), Some("first"));
        assert_eq!(map.get(&2).map(String::as_str), Some("third"));
        assert!(!map.contains_key(&3));
    }
}
