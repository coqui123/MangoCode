//! Main-buffer hybrid TUI.
//!
//! The transcript is real terminal output, so terminal-native scrollback works.
//! The live composer is rendered from ratatui widgets into an offscreen buffer
//! and repainted in reserved bottom rows.

use std::{
    borrow::Cow,
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    hash::{Hash, Hasher},
    io::{self, Stdout, Write},
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
    messages::{render_message, render_structured_tool_result, RenderContext},
    prompt_input::{input_height, render_prompt_input, InputMode},
    render::{
        build_tool_names, compact_model_label, flush_sixel_blit_with_cursor,
        greeting_footer_lines, greeting_header_lines, greeting_scrollback_height,
        greeting_scrollback_lines, mascot_rect, render_app, reset_last_sixel_position,
        reset_sixel_blit_state,
    },
};

const MANGO: RtColor = RtColor::Rgb(250, 114, 145); // #FA7291 light rose (mascot)
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
    rendered_structured_tool_outputs: HashSet<String>,
    expansion_fingerprint: u64,
    deferred_new_messages: bool,
    last_terminal_size: Option<(u16, u16)>,
    /// Cell rect (y, x, w, h) the mascot overlay occupied in the last fullscreen
    /// frame, or `None` if it wasn't drawn. Used to detect when the mascot is
    /// hidden or moved (e.g. via `/greeting`) so the stale image can be cleared
    /// instead of leaving a ghost the next blit overlaps.
    last_mascot_rect: Option<(u16, u16, u16, u16)>,
    /// Whether the greeting/welcome box has been printed at the top of the
    /// current transcript. Reset whenever the transcript is cleared so it's
    /// reprinted when the conversation is rebuilt. Honours `app.show_greeting`.
    greeting_printed: bool,
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
            rendered_structured_tool_outputs: HashSet::new(),
            expansion_fingerprint: expansion_fingerprint(app),
            deferred_new_messages: false,
            last_terminal_size: None,
            last_mascot_rect: None,
            greeting_printed: false,
            restored: false,
        };
        terminal.render_live(app)?;
        Ok(terminal)
    }

    /// Return cached terminal size, falling back to a fresh `terminal::size()`
    /// query if no size has been recorded yet. Updated by `render_live` and
    /// `render_fullscreen_app` at the top of each frame, so internal helpers
    /// can avoid one syscall per call.
    #[inline]
    fn current_size(&self) -> (u16, u16) {
        self.last_terminal_size
            .unwrap_or_else(|| terminal::size().unwrap_or((80, 24)))
    }

    /// Force a fresh size poll and update the cache. The CLI calls this from
    /// its `Event::Resize` handler so any code path that runs between the
    /// resize and the next `render_live` (e.g. an async query event handler)
    /// sees the new dimensions instead of a stale cached value.
    pub fn invalidate_size_cache(&mut self) {
        self.last_terminal_size = Some(terminal::size().unwrap_or((80, 24)));
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
        // Transcript is being rebuilt from scratch — the greeting (if any) must
        // be reprinted at the top.
        self.greeting_printed = false;
        if app.active_modal_owner().is_some() || messages.is_empty() {
            self.printed_messages = 0;
            self.transcript_row = 0;
            self.stream_open = false;
            self.composer_buffer = None;
            self.fullscreen_buffer = None;
            self.screen_lines.clear();
            self.rendered_structured_tool_outputs.clear();
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
        // Evict accumulated Kitty graphics images from terminal memory so
        // long sessions don't keep every transcript image cached forever.
        // No-op on terminals that don't support the Kitty protocol.
        if matches!(
            crate::kitty_image::detect_image_protocol(),
            crate::kitty_image::ImageProtocol::Kitty
        ) {
            crate::kitty_image::emit_kitty_delete_all();
            // Terminal-side image store is now empty, so the in-process
            // dedup cache must follow — otherwise we'd skip uploads on
            // hash hits for images the terminal no longer has.
            crate::kitty_image::reset_transcript_image_cache();
        }
        self.composer_height = 0;
        self.transcript_row = 0;
        self.stream_open = false;
        self.printed_messages = 0;
        self.composer_buffer = None;
        self.fullscreen_buffer = None;
        self.screen_lines.clear();
        self.rendered_structured_tool_outputs.clear();
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
            self.last_terminal_size = Some((cols, rows));
            return Ok(());
        }
        if self.overlay_active {
            self.leave_fullscreen(false)?;
        }
        if let Some((old_cols, old_rows)) = self.last_terminal_size {
            if old_cols != cols || old_rows != rows {
                self.clear_composer_area_at(old_rows, self.composer_height)?;
                self.composer_buffer = None;
                if old_rows != rows {
                    self.reset_scroll_region()?;
                    self.apply_scroll_region()?;
                }
            }
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
        self.last_terminal_size = Some((cols, rows));
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
        let (cols, _) = self.current_size();
        let ctx = RenderContext {
            width: cols.saturating_sub(1).max(1),
            tool_names: build_tool_names(&app.messages),
            expanded_thinking: app.thinking_expanded.clone(),
            expanded_tool_outputs: app.expanded_tool_outputs.clone(),
            ..Default::default()
        };
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
                metadata,
                ..
            } => {
                self.move_to_transcript_end()?;
                if !*is_error {
                    let width = self.current_size().0.saturating_sub(1).max(1);
                    if let Some(rendered) = render_structured_tool_result(metadata.as_ref(), width)
                    {
                        for line in rendered {
                            self.write_line_with_tool(&line, Some(tool_id.as_str()), Some(app))?;
                        }
                        self.rendered_structured_tool_outputs
                            .insert(tool_id.clone());
                        self.write_blank_line_for_app(app)?;
                        return self.render_live(app);
                    }
                }
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
            self.rendered_structured_tool_outputs.clear();
            // Leaving the welcome overlay for scrollback: the greeting must be
            // reprinted as the first transcript content so it persists.
            self.greeting_printed = false;
        }
        // Print the greeting at the top of the transcript before any message.
        self.ensure_greeting_header(app)?;
        Ok(())
    }

    /// Print the greeting/welcome box at the top of scrollback once per
    /// transcript build, so it stays visible after the first prompt instead of
    /// vanishing with the welcome overlay. No-op when `/greeting` is off.
    fn ensure_greeting_header(&mut self, app: &App) -> io::Result<()> {
        if self.greeting_printed || !app.show_greeting || app.active_modal_owner().is_some() {
            return Ok(());
        }
        // Set the guard before writing so the nested write helpers (which call
        // back into transcript bookkeeping) can't recurse into this.
        self.greeting_printed = true;
        self.print_greeting_block(app)
    }

    fn print_greeting_block(&mut self, app: &App) -> io::Result<()> {
        use crate::kitty_image::{detect_image_protocol, ImageProtocol};
        let (cols, rows) = self.current_size();

        // Preferred path: emit the real SVG mascot inline so it persists in
        // scrollback and scrolls with the transcript. Sized to an exact row
        // count (via the terminal's reported cell height) so the manual row
        // counter stays in sync with the rendered image.
        let image_capable = app.mascot_sixel.is_some()
            && !matches!(detect_image_protocol(), ImageProtocol::Text);
        if image_capable {
            let want_rows = rows.saturating_sub(10).clamp(4, 10);
            let cell_px_h = crossterm::terminal::window_size()
                .ok()
                .filter(|ws| ws.rows != 0 && ws.height != 0)
                .map(|ws| ws.height as f32 / ws.rows as f32)
                .unwrap_or(16.0);
            if let Some((escape, img_rows)) =
                crate::rustle::encode_mascot_inline_for_rows(want_rows, cell_px_h)
            {
                for line in greeting_header_lines(app) {
                    self.write_styled_line_for_app(line.spans, app)?;
                }
                self.print_mascot_image(&escape, img_rows)?;
                for line in greeting_footer_lines(app) {
                    self.write_styled_line_for_app(line.spans, app)?;
                }
                self.write_blank_line_for_app(app)?;
                return Ok(());
            }
        }

        // Fallback (no image protocol or rasterization failed): text-art mascot.
        let width = cols.saturating_sub(1).max(20);
        let height = greeting_scrollback_height(rows);
        let lines = greeting_scrollback_lines(app, width, height);
        if lines.is_empty() {
            return Ok(());
        }
        for line in lines {
            self.write_styled_line_for_app(line.spans, app)?;
        }
        self.write_blank_line_for_app(app)?;
        Ok(())
    }

    /// Emit the mascot inline image at the current transcript position and
    /// reserve `img_rows` rows for it, leaving the cursor below the image so the
    /// next printed line doesn't overwrite it. The image becomes part of
    /// scrollback and scrolls with the transcript.
    fn print_mascot_image(&mut self, escape: &str, img_rows: u16) -> io::Result<()> {
        self.move_to_transcript_end()?;
        self.composer_buffer = None;
        queue_base_style(&mut self.stdout)?;
        // Image protocols place the image with its top-left at the cursor.
        write!(self.stdout, "{}", escape)?;
        let (_, rows) = self.current_size();
        let bottom = rows.saturating_sub(self.composer_height).saturating_sub(1);
        self.transcript_row = self
            .transcript_row
            .saturating_add(img_rows.max(1))
            .min(bottom);
        queue!(self.stdout, MoveTo(0, self.transcript_row))?;
        self.stdout.flush()
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
            if let Some(msg) = live_transcript_message(msg, &self.rendered_structured_tool_outputs)
            {
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
        let (_, rows) = self.current_size();
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
        let (_, rows) = self.current_size();
        let bottom = rows.saturating_sub(self.composer_height).saturating_sub(1);
        self.transcript_row = self.transcript_row.min(bottom);
        queue!(self.stdout, MoveTo(0, self.transcript_row))?;
        self.stdout.flush()
    }

    fn clear_composer_area(&mut self) -> io::Result<()> {
        if self.composer_height == 0 {
            return Ok(());
        }
        let (_, rows) = self.current_size();
        self.clear_composer_area_at(rows, self.composer_height)
    }

    fn clear_composer_area_at(&mut self, rows: u16, composer_height: u16) -> io::Result<()> {
        if composer_height == 0 {
            return Ok(());
        }
        let top = rows.saturating_sub(composer_height);
        let (_, current_rows) = self.current_size();
        if top >= current_rows {
            return Ok(());
        }
        for i in 0..composer_height {
            let row = top.saturating_add(i);
            if row >= current_rows {
                break;
            }
            queue_base_style(&mut self.stdout)?;
            queue!(self.stdout, MoveTo(0, row), Clear(ClearType::CurrentLine))?;
        }
        self.stdout.flush()
    }

    fn flush_buffer(&mut self, screen_y: u16, buffer: &Buffer) -> io::Result<()> {
        let (cols, rows) = self.current_size();
        for y in 0..buffer.area.height {
            queue!(self.stdout, MoveTo(0, screen_y.saturating_add(y)))?;
            for x in 0..buffer.area.width {
                if screen_y.saturating_add(y) == rows.saturating_sub(1)
                    && x == cols.saturating_sub(1)
                {
                    continue;
                }
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

        let (cols, rows) = self.current_size();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                if screen_y.saturating_add(y) == rows.saturating_sub(1)
                    && x == cols.saturating_sub(1)
                {
                    continue;
                }
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
        let (_, rows) = self.current_size();
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
        let width = self.current_size().0 as usize;
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

        // When the mascot is hidden or moved (e.g. `/greeting` toggles the
        // welcome box), its cells are blank in both the old and new buffers, so
        // a diff redraw never repaints them and the inline image lingers as a
        // ghost — a re-blit then overlaps a second mascot. Detect the change and
        // force a full repaint so those cells get rewritten (erasing the Sixel),
        // plus delete the stale Kitty placement which lives outside the grid.
        let mascot_now = mascot_rect();
        let mascot_layout_changed = mascot_now != self.last_mascot_rect;
        if mascot_layout_changed {
            clear_stale_mascot_image();
        }

        let needs_full_flush = mascot_layout_changed
            || previous
                .as_ref()
                .is_none_or(|prev| prev.area != buffer.area);
        // A text selection / Ctrl+C copy makes the terminal repaint the cells
        // under the mascot, wiping the inline image overlay (the image is not
        // part of the cell grid). Detect when this redraw touches the mascot
        // region and force a re-blit — otherwise the position guard suppresses
        // re-emission and the mascot stays gone.
        let mascot_disturbed = !needs_full_flush
            && previous
                .as_ref()
                .is_some_and(|prev| mascot_region_changed(prev, &buffer));
        // Force a full repaint (previous = None) on a mascot layout change so the
        // old image's cells are rewritten rather than diffed away.
        let diff_base = if mascot_layout_changed {
            None
        } else {
            previous.as_ref()
        };
        self.flush_buffer_diff(0, &buffer, diff_base)?;
        self.fullscreen_buffer = Some(buffer);
        self.last_mascot_rect = mascot_now;
        queue!(self.stdout, MoveTo(cursor.x, cursor.y))?;
        self.stdout.flush()?;
        if needs_full_flush || mascot_disturbed {
            reset_last_sixel_position();
        }
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
    let model_short = compact_model_label(&app.model_name, app.config.provider.as_deref());
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

fn live_transcript_message(
    msg: &Message,
    rendered_structured_tool_outputs: &HashSet<String>,
) -> Option<Message> {
    match &msg.content {
        MessageContent::Text(_) => Some(msg.clone()),
        MessageContent::Blocks(blocks) => {
            let visible = blocks
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolUse { .. } => false,
                    ContentBlock::ToolResult {
                        tool_use_id,
                        metadata,
                        ..
                    } => {
                        structured_tool_result_kind(metadata.as_ref()).is_some()
                            && !rendered_structured_tool_outputs.contains(tool_use_id)
                    }
                    _ => true,
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

fn structured_tool_result_kind(metadata: Option<&serde_json::Value>) -> Option<&str> {
    let kind = metadata?.get("transcript_display")?.get("kind")?.as_str()?;
    match kind {
        "updated_plan" | "file_changes" => Some(kind),
        _ => None,
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

/// Remove a stale mascot image before the screen is repainted at a new layout.
/// For Sixel/iTerm the pixels live in the cell grid and are cleared by the
/// forced full repaint of the caller; for Kitty the image floats above the grid
/// and must be deleted explicitly. Also drops the position guard so the mascot
/// re-blits cleanly if it's still shown.
fn clear_stale_mascot_image() {
    crate::kitty_image::delete_mascot_placement();
    reset_last_sixel_position();
}

/// Whether any cell inside the mascot image's rectangle differs between the
/// previously-flushed buffer and the new one. A difference means the terminal
/// repainted those cells (e.g. drawing/clearing a selection highlight) and thus
/// wiped the inline image overlay, so the caller must force a re-blit.
fn mascot_region_changed(prev: &Buffer, next: &Buffer) -> bool {
    let Some((y, x, w, h)) = mascot_rect() else {
        return false;
    };
    if prev.area != next.area {
        return true;
    }
    let area = next.area;
    let x_end = x.saturating_add(w).min(area.x + area.width);
    let y_end = y.saturating_add(h).min(area.y + area.height);
    for cy in y..y_end {
        for cx in x..x_end {
            if cx >= area.x + area.width || cy >= area.y + area.height {
                continue;
            }
            let n = &next[(cx, cy)];
            let p = &prev[(cx, cy)];
            if n.symbol() != p.symbol() || n.style() != p.style() {
                return true;
            }
        }
    }
    false
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

fn truncate(text: &str, max_chars: usize) -> Cow<'_, str> {
    if text.chars().count() <= max_chars {
        Cow::Borrowed(text)
    } else {
        let mut s: String = text.chars().take(max_chars).collect();
        s.push('…');
        Cow::Owned(s)
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
                metadata: None,
            },
            ContentBlock::Text {
                text: "done".to_string(),
            },
        ]);

        let visible = live_transcript_message(&msg, &HashSet::new())
            .expect("final text should remain visible");
        assert_eq!(visible.content_blocks().len(), 1);
        assert_eq!(visible.get_all_text(), "done");
    }

    #[test]
    fn live_transcript_message_keeps_unrendered_structured_tool_result() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "tool-1".to_string(),
            content: ToolResultContent::Text("Plan updated".to_string()),
            is_error: Some(false),
            metadata: Some(serde_json::json!({
                "transcript_display": {
                    "kind": "updated_plan",
                    "plan": []
                }
            })),
        }]);

        let visible = live_transcript_message(&msg, &HashSet::new())
            .expect("unrendered structured tool result should be replayed");
        assert_eq!(visible.content_blocks().len(), 1);
    }

    #[test]
    fn live_transcript_message_skips_structured_tool_result_already_rendered_live() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "tool-1".to_string(),
            content: ToolResultContent::Text("Plan updated".to_string()),
            is_error: Some(false),
            metadata: Some(serde_json::json!({
                "transcript_display": {
                    "kind": "updated_plan",
                    "plan": []
                }
            })),
        }]);
        let rendered = HashSet::from(["tool-1".to_string()]);

        assert!(live_transcript_message(&msg, &rendered).is_none());
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
