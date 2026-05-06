//! Diff viewer TUI component.
//!
//! Shows a two-pane diff dialog: file list (left) + unified diff detail (right).
//! Keyboard: ↑↓ navigate files, Tab switch pane, t toggle diff type, Esc close.

use mangocode_core::file_history::FileHistory;
use mangocode_turn_diff::patches_for_turn;
use once_cell::sync::Lazy;
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;

static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEME_SET: Lazy<ThemeSet> = Lazy::new(ThemeSet::load_defaults);

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single hunk of a unified diff.
#[derive(Debug, Clone)]
pub struct DiffHunk {
    /// Original line number range: (start, count).
    pub old_range: (u32, u32),
    /// New line number range: (start, count).
    pub new_range: (u32, u32),
    /// Lines in this hunk.
    pub lines: Vec<DiffLine>,
}

/// A single line in a diff hunk.
#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub content: String,
    /// Original line number (if applicable).
    pub old_line_no: Option<u32>,
    /// New line number (if applicable).
    pub new_line_no: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffLineKind {
    /// Unchanged context line.
    Context,
    /// Added line.
    Added,
    /// Removed line.
    Removed,
    /// Hunk header (@@ line).
    Header,
}

/// Stats for a single file in the diff.
#[derive(Debug, Clone)]
pub struct FileDiffStats {
    /// File path (relative to project root).
    pub path: String,
    /// Number of added lines.
    pub added: u32,
    /// Number of removed lines.
    pub removed: u32,
    /// Is this a binary file?
    pub binary: bool,
    /// Is this a newly created file (no previous version)?
    pub is_new_file: bool,
    /// Is this a deleted file (no current version)?
    pub is_deleted_file: bool,
    /// Stable patch id assigned for turn-scoped changes.
    pub patch_id: Option<String>,
    /// Conversation turn index that produced this patch.
    pub turn_index: Option<usize>,
    /// Tool names that contributed to this patch.
    pub tool_names: Vec<String>,
    /// All hunks for this file.
    pub hunks: Vec<DiffHunk>,
}

/// Which diff type to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// `git diff` since last commit.
    GitDiff,
    /// Changes made during this conversation turn.
    TurnDiff,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Active pane in the diff dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffPane {
    FileList,
    Detail,
}

/// Full state for the diff viewer dialog.
#[derive(Debug, Clone)]
pub struct DiffViewerState {
    /// All files in the diff.
    pub files: Vec<FileDiffStats>,
    /// Cached turn-specific files, populated externally.
    pub turn_files: Vec<FileDiffStats>,
    /// Currently selected file index.
    pub selected_file: usize,
    /// Active pane.
    pub active_pane: DiffPane,
    /// Current diff type.
    pub diff_type: DiffType,
    /// Scroll offset for the detail pane (in lines).
    pub detail_scroll: u16,
    /// Rendered line cache: (file_index, terminal_width) → lines.
    render_cache: HashMap<(usize, u16), Vec<String>>,
    /// Whether the dialog is open.
    pub open: bool,
    /// Per-file collapsed state (indexed by file position in `files`).
    pub collapsed: Vec<bool>,
}

impl DiffViewerState {
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            turn_files: Vec::new(),
            selected_file: 0,
            active_pane: DiffPane::FileList,
            diff_type: DiffType::GitDiff,
            detail_scroll: 0,
            render_cache: HashMap::new(),
            open: false,
            collapsed: Vec::new(),
        }
    }

    /// Toggle collapsed state for the currently selected file.
    pub fn toggle_file_collapse(&mut self) {
        if let Some(c) = self.collapsed.get_mut(self.selected_file) {
            *c = !*c;
            self.detail_scroll = 0;
        }
    }

    /// Open the dialog and load diffs from the project root.
    pub fn open(&mut self, project_root: &std::path::Path) {
        self.open_for_type(DiffType::GitDiff, project_root);
    }

    /// Open directly in turn-diff mode.
    pub fn open_turn(&mut self, project_root: &std::path::Path) {
        self.open_for_type(DiffType::TurnDiff, project_root);
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn select_prev(&mut self) {
        if self.selected_file > 0 {
            self.selected_file -= 1;
            self.detail_scroll = 0;
        }
    }

    pub fn select_next(&mut self) {
        if self.selected_file + 1 < self.files.len() {
            self.selected_file += 1;
            self.detail_scroll = 0;
        }
    }

    pub fn switch_pane(&mut self) {
        self.active_pane = match self.active_pane {
            DiffPane::FileList => DiffPane::Detail,
            DiffPane::Detail => DiffPane::FileList,
        };
    }

    pub fn toggle_diff_type(&mut self, project_root: &std::path::Path) {
        self.diff_type = match self.diff_type {
            DiffType::GitDiff => DiffType::TurnDiff,
            DiffType::TurnDiff => DiffType::GitDiff,
        };
        self.reload_files(project_root);
    }

    pub fn scroll_detail_up(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_sub(3);
    }

    pub fn scroll_detail_down(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_add(3);
    }

    pub fn set_turn_diff(&mut self, files: Vec<FileDiffStats>) {
        self.turn_files = files;
        if self.diff_type == DiffType::TurnDiff {
            self.files = self.turn_files.clone();
            self.selected_file = 0;
            self.detail_scroll = 0;
            self.render_cache.clear();
            self.collapsed = vec![false; self.files.len()];
        }
    }

    fn open_for_type(&mut self, diff_type: DiffType, project_root: &std::path::Path) {
        self.diff_type = diff_type;
        self.reload_files(project_root);
        self.open = true;
    }

    fn reload_files(&mut self, project_root: &std::path::Path) {
        self.files = match self.diff_type {
            DiffType::GitDiff => load_git_diff(project_root),
            DiffType::TurnDiff => self.turn_files.clone(),
        };
        self.selected_file = 0;
        self.detail_scroll = 0;
        self.render_cache.clear();
        self.collapsed = vec![false; self.files.len()];
    }
}

impl Default for DiffViewerState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Load the current `git diff HEAD` from `project_root`.
pub fn load_git_diff(project_root: &std::path::Path) -> Vec<FileDiffStats> {
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD", "--unified=3"])
        .current_dir(project_root)
        .output();

    let text = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        Ok(_out) => {
            // Try just `git diff` (no HEAD) for unstaged changes
            let out2 = std::process::Command::new("git")
                .args(["diff", "--unified=3"])
                .current_dir(project_root)
                .output();
            match out2 {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                _ => return Vec::new(),
            }
        }
        Err(_) => return Vec::new(),
    };

    parse_unified_diff(&text)
}

/// Build a turn-local diff from file-history snapshots.
pub fn build_turn_diff(
    file_history: &FileHistory,
    turn_index: usize,
    project_root: &std::path::Path,
) -> Vec<FileDiffStats> {
    patches_for_turn(file_history, turn_index, project_root)
        .into_iter()
        .filter_map(|patch| {
            let is_new_file = !patch.before_exists && patch.after_exists;
            let is_deleted_file = patch.before_exists && !patch.after_exists;
            let mut file = if patch.binary {
                FileDiffStats {
                    path: patch.relative_path.clone(),
                    added: 0,
                    removed: 0,
                    binary: true,
                    is_new_file,
                    is_deleted_file,
                    patch_id: None,
                    turn_index: None,
                    tool_names: Vec::new(),
                    hunks: Vec::new(),
                }
            } else {
                let before = patch.before_text.as_deref().unwrap_or("");
                let after = patch.after_text.as_deref().unwrap_or("");
                build_file_diff_from_snapshots(patch.relative_path.clone(), before, after)
            };

            file.is_new_file = is_new_file;
            file.is_deleted_file = is_deleted_file;
            file.patch_id = Some(patch.id.0);
            file.turn_index = Some(patch.turn_index);
            file.tool_names = patch.tool_names;

            if file.binary || !file.hunks.is_empty() || is_new_file || is_deleted_file {
                Some(file)
            } else {
                None
            }
        })
        .collect()
}

pub fn build_latest_turn_diff(
    file_history: &FileHistory,
    project_root: &std::path::Path,
) -> Vec<FileDiffStats> {
    let Some(turn_index) = file_history.latest_turn_index() else {
        return Vec::new();
    };
    build_turn_diff(file_history, turn_index, project_root)
}

fn build_file_diff_from_snapshots(path: String, before: &str, after: &str) -> FileDiffStats {
    let diff = TextDiff::from_lines(before, after);
    let mut added = 0u32;
    let mut removed = 0u32;
    let mut hunks = Vec::new();

    for group in diff.grouped_ops(3) {
        let mut lines = Vec::new();

        for op in group {
            for change in diff.iter_changes(&op) {
                let mut content = change.to_string();
                if content.ends_with('\n') {
                    content.pop();
                    if content.ends_with('\r') {
                        content.pop();
                    }
                }

                let kind = match change.tag() {
                    ChangeTag::Equal => DiffLineKind::Context,
                    ChangeTag::Delete => {
                        removed += 1;
                        DiffLineKind::Removed
                    }
                    ChangeTag::Insert => {
                        added += 1;
                        DiffLineKind::Added
                    }
                };

                lines.push(DiffLine {
                    kind,
                    content,
                    old_line_no: change.old_index().map(|idx| idx as u32 + 1),
                    new_line_no: change.new_index().map(|idx| idx as u32 + 1),
                });
            }
        }

        let old_range = summarize_old_range(&lines);
        let new_range = summarize_new_range(&lines);
        lines.insert(
            0,
            DiffLine {
                kind: DiffLineKind::Header,
                content: format!(
                    "@@ -{},{} +{},{} @@",
                    old_range.0, old_range.1, new_range.0, new_range.1
                ),
                old_line_no: None,
                new_line_no: None,
            },
        );

        hunks.push(DiffHunk {
            old_range,
            new_range,
            lines,
        });
    }

    FileDiffStats {
        path,
        added,
        removed,
        binary: false,
        is_new_file: before.is_empty() && !after.is_empty(),
        is_deleted_file: !before.is_empty() && after.is_empty(),
        patch_id: None,
        turn_index: None,
        tool_names: Vec::new(),
        hunks,
    }
}

fn summarize_old_range(lines: &[DiffLine]) -> (u32, u32) {
    summarize_range(lines.iter().filter_map(|line| line.old_line_no).collect())
}

fn summarize_new_range(lines: &[DiffLine]) -> (u32, u32) {
    summarize_range(lines.iter().filter_map(|line| line.new_line_no).collect())
}

fn summarize_range(line_numbers: Vec<u32>) -> (u32, u32) {
    match (line_numbers.first().copied(), line_numbers.last().copied()) {
        (Some(start), Some(end)) => (start, end.saturating_sub(start) + 1),
        _ => (0, 0),
    }
}

/// Parse unified diff text into `Vec<FileDiffStats>`.
pub fn parse_unified_diff(text: &str) -> Vec<FileDiffStats> {
    let mut files: Vec<FileDiffStats> = Vec::new();
    let mut current_file: Option<FileDiffStats> = None;
    let mut current_hunk: Option<DiffHunk> = None;
    let mut old_line = 0u32;
    let mut new_line = 0u32;

    for raw_line in text.lines() {
        if raw_line.starts_with("diff --git ") {
            // Flush previous hunk and file
            if let Some(hunk) = current_hunk.take() {
                if let Some(f) = current_file.as_mut() {
                    f.hunks.push(hunk);
                }
            }
            if let Some(f) = current_file.take() {
                files.push(f);
            }
            // Extract file path from "diff --git a/foo b/foo"
            let path = raw_line
                .split_whitespace()
                .nth(3)
                .map(|s| s.strip_prefix("b/").unwrap_or(s).to_string())
                .unwrap_or_else(|| "unknown".to_string());
            current_file = Some(FileDiffStats {
                path,
                added: 0,
                removed: 0,
                binary: false,
                is_new_file: false,
                is_deleted_file: false,
                patch_id: None,
                turn_index: None,
                tool_names: Vec::new(),
                hunks: Vec::new(),
            });
        } else if raw_line.starts_with("new file mode") {
            if let Some(f) = current_file.as_mut() {
                f.is_new_file = true;
            }
        } else if raw_line.starts_with("deleted file mode") {
            if let Some(f) = current_file.as_mut() {
                f.is_deleted_file = true;
            }
        } else if raw_line.starts_with("Binary files ") {
            if let Some(f) = current_file.as_mut() {
                f.binary = true;
            }
        } else if raw_line.starts_with("@@ ") {
            // Flush previous hunk
            if let Some(hunk) = current_hunk.take() {
                if let Some(f) = current_file.as_mut() {
                    f.hunks.push(hunk);
                }
            }
            // Parse @@ -old_start,old_count +new_start,new_count @@
            let (old_start, old_count, new_start, new_count) = parse_hunk_header(raw_line);
            old_line = old_start;
            new_line = new_start;
            current_hunk = Some(DiffHunk {
                old_range: (old_start, old_count),
                new_range: (new_start, new_count),
                lines: vec![DiffLine {
                    kind: DiffLineKind::Header,
                    content: raw_line.to_string(),
                    old_line_no: None,
                    new_line_no: None,
                }],
            });
        } else if let Some(hunk) = current_hunk.as_mut() {
            if raw_line.starts_with('+') && !raw_line.starts_with("+++") {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Added,
                    content: raw_line[1..].to_string(),
                    old_line_no: None,
                    new_line_no: Some(new_line),
                });
                new_line += 1;
                if let Some(f) = current_file.as_mut() {
                    f.added += 1;
                }
            } else if raw_line.starts_with('-') && !raw_line.starts_with("---") {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Removed,
                    content: raw_line[1..].to_string(),
                    old_line_no: Some(old_line),
                    new_line_no: None,
                });
                old_line += 1;
                if let Some(f) = current_file.as_mut() {
                    f.removed += 1;
                }
            } else if let Some(stripped) = raw_line.strip_prefix(' ') {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Context,
                    content: stripped.to_string(),
                    old_line_no: Some(old_line),
                    new_line_no: Some(new_line),
                });
                old_line += 1;
                new_line += 1;
            }
        }
    }

    // Flush final hunk and file
    if let Some(hunk) = current_hunk.take() {
        if let Some(f) = current_file.as_mut() {
            f.hunks.push(hunk);
        }
    }
    if let Some(f) = current_file.take() {
        files.push(f);
    }

    files
}

fn parse_hunk_header(line: &str) -> (u32, u32, u32, u32) {
    // @@ -old_start,old_count +new_start,new_count @@
    let parts: Vec<&str> = line.split_whitespace().collect();
    let parse_range = |s: &str| -> (u32, u32) {
        let s = s.trim_start_matches(['-', '+']);
        if let Some(comma) = s.find(',') {
            let start = s[..comma].parse().unwrap_or(1);
            let count = s[comma + 1..].parse().unwrap_or(0);
            (start, count)
        } else {
            (s.parse().unwrap_or(1), 1)
        }
    };
    let old = parts.get(1).map(|s| parse_range(s)).unwrap_or((1, 0));
    let new = parts.get(2).map(|s| parse_range(s)).unwrap_or((1, 0));
    (old.0, old.1, new.0, new.1)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the diff dialog overlay.
pub fn render_diff_dialog(state: &mut DiffViewerState, area: Rect, buf: &mut Buffer) {
    if !state.open {
        return;
    }

    // Center the dialog (80% width, 80% height)
    let dialog_width = (area.width * 4 / 5).max(40).min(area.width);
    let dialog_height = (area.height * 4 / 5).max(10).min(area.height);
    let x = area.x + (area.width - dialog_width) / 2;
    let y = area.y + (area.height - dialog_height) / 2;
    let dialog_area = Rect {
        x,
        y,
        width: dialog_width,
        height: dialog_height,
    };

    // Clear the area
    Clear.render(dialog_area, buf);

    // Outer border
    let title = match state.diff_type {
        DiffType::GitDiff => " Diff (git) [Space: collapse, d: toggle, Tab: pane, Esc: close] ",
        DiffType::TurnDiff => " Diff (turn) [Space: collapse, d: toggle, Tab: pane, Esc: close] ",
    };
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::White))
        .render(dialog_area, buf);

    let inner = Rect {
        x: dialog_area.x + 1,
        y: dialog_area.y + 1,
        width: dialog_area.width.saturating_sub(2),
        height: dialog_area.height.saturating_sub(2),
    };

    if state.files.is_empty() {
        let empty = match state.diff_type {
            DiffType::GitDiff => "No git changes available.",
            DiffType::TurnDiff => "No turn changes available yet.",
        };
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                empty,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "No tracked file changes were captured for the selected turn.",
                Style::default().fg(Color::DarkGray),
            )]),
        ])
        .render(inner, buf);
        return;
    }

    // Split: file list 30%, detail 70%
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(inner);

    render_file_list(state, panes[0], buf);
    render_diff_detail(state, panes[1], buf);
}

fn short_patch_id(id: &str) -> String {
    let suffix = id.rsplit('-').next().unwrap_or(id);
    format!("#{}", suffix.chars().take(8).collect::<String>())
}

fn truncate_path_tail(path: &str, max_chars: usize) -> String {
    if path.chars().count() <= max_chars {
        return path.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    if max_chars == 1 {
        return "…".to_string();
    }

    let tail = path
        .chars()
        .rev()
        .take(max_chars - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("…{tail}")
}

fn render_file_list(state: &DiffViewerState, area: Rect, buf: &mut Buffer) {
    let focused = state.active_pane == DiffPane::FileList;
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .title(" Files ")
        .borders(Borders::ALL)
        .border_style(border_style)
        .render(area, buf);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    let max_visible = inner.height as usize;
    let start = state.selected_file.saturating_sub(max_visible / 2);
    let end = (start + max_visible).min(state.files.len());

    for (i, file) in state.files[start..end].iter().enumerate() {
        let abs_idx = start + i;
        let selected = abs_idx == state.selected_file;

        // Truncate path to fit
        let patch_badge = file
            .patch_id
            .as_ref()
            .map(|id| format!(" {}", short_patch_id(id)))
            .unwrap_or_default();
        let avail = inner.width.saturating_sub(10 + patch_badge.len() as u16) as usize;
        let path = truncate_path_tail(&file.path, avail);

        let is_collapsed = *state.collapsed.get(abs_idx).unwrap_or(&false);
        let collapse_char = if is_collapsed { "\u{25b8}" } else { "\u{25be}" }; // ▸ / ▾
        let (stats, stats_color) = if file.binary {
            ("[binary]".to_string(), Color::DarkGray)
        } else if file.is_new_file {
            (format!("[new] +{}", file.added), Color::Yellow)
        } else if file.is_deleted_file {
            (format!("[deleted] -{}", file.removed), Color::Red)
        } else {
            (
                format!("+{} -{}", file.added, file.removed),
                Color::DarkGray,
            )
        };

        let base_style = if selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        // Render prefix + path
        let y = inner.y + i as u16;
        if y >= area.y + area.height {
            break;
        }

        let collapse_style = Style::default().fg(Color::DarkGray);
        let line = Line::from(vec![
            Span::styled(if selected { "> " } else { "  " }, base_style),
            Span::styled(format!("{} ", collapse_char), collapse_style),
            Span::styled(
                path,
                base_style.fg(if selected { Color::White } else { Color::Gray }),
            ),
            Span::styled(patch_badge, Style::default().fg(Color::Cyan)),
        ]);
        let row_area = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: 1,
        };
        Paragraph::new(line).render(row_area, buf);

        // Stats on the right side
        let stats_x = inner.x + inner.width.saturating_sub(stats.len() as u16 + 1);
        if stats_x > inner.x {
            let stats_area = Rect {
                x: stats_x,
                y,
                width: stats.len() as u16,
                height: 1,
            };
            Paragraph::new(Line::from(vec![Span::styled(
                stats,
                Style::default().fg(stats_color),
            )]))
            .render(stats_area, buf);
        }
    }

    // Pagination indicators
    if start > 0 {
        let ind_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: 1,
        };
        Paragraph::new(format!("↑ {} more", start))
            .style(Style::default().fg(Color::DarkGray))
            .render(ind_area, buf);
    }
    if end < state.files.len() {
        let remaining = state.files.len() - end;
        let y = inner.y + inner.height.saturating_sub(1);
        let ind_area = Rect {
            x: inner.x,
            y,
            width: inner.width,
            height: 1,
        };
        Paragraph::new(format!("↓ {} more", remaining))
            .style(Style::default().fg(Color::DarkGray))
            .render(ind_area, buf);
    }
}

fn render_diff_detail(state: &DiffViewerState, area: Rect, buf: &mut Buffer) {
    let focused = state.active_pane == DiffPane::Detail;
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let file = match state.files.get(state.selected_file) {
        Some(f) => f,
        None => return,
    };

    let title = match (&file.patch_id, file.turn_index) {
        (Some(id), Some(turn)) => format!(" {} [{} turn {}] ", file.path, id, turn),
        (Some(id), None) => format!(" {} [{}] ", file.path, id),
        _ => format!(" {} ", file.path),
    };
    Block::default()
        .title(title.as_str())
        .borders(Borders::ALL)
        .border_style(border_style)
        .render(area, buf);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    if *state.collapsed.get(state.selected_file).unwrap_or(&false) {
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "  [collapsed]  press Space to expand",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]),
        ])
        .render(inner, buf);
        return;
    }

    if file.binary {
        Paragraph::new("Binary file — no diff available")
            .style(Style::default().fg(Color::DarkGray))
            .render(inner, buf);
        return;
    }

    // Build lines for rendering
    let lines = build_diff_lines(file, inner.width);
    let total_lines = lines.len();
    let scroll =
        (state.detail_scroll as usize).min(total_lines.saturating_sub(inner.height as usize));
    let visible = &lines[scroll..];

    // Shrink inner width by 1 to leave room for scrollbar
    let text_width = if total_lines > inner.height as usize {
        inner.width.saturating_sub(1)
    } else {
        inner.width
    };

    for (i, line) in visible.iter().enumerate() {
        if i as u16 >= inner.height {
            break;
        }
        let y = inner.y + i as u16;
        let row_area = Rect {
            x: inner.x,
            y,
            width: text_width,
            height: 1,
        };
        Paragraph::new(line.clone()).render(row_area, buf);
    }

    // Simple scrollbar on the rightmost column of inner
    if total_lines > inner.height as usize && inner.width > 1 {
        let bar_x = inner.x + inner.width - 1;
        let bar_h = inner.height as usize;
        // Thumb size proportional to visible fraction, minimum 1
        let thumb_size = ((bar_h * bar_h) / total_lines).max(1).min(bar_h);
        // Thumb position
        let scroll_range = total_lines.saturating_sub(bar_h);
        let thumb_top = if scroll_range > 0 {
            (scroll * (bar_h.saturating_sub(thumb_size))) / scroll_range
        } else {
            0
        };

        for row in 0..bar_h {
            let y = inner.y + row as u16;
            let ch = if row == 0 {
                '\u{25b2}' // ▲
            } else if row == bar_h - 1 {
                '\u{25bc}' // ▼
            } else if row > thumb_top && row < thumb_top + thumb_size + 1 {
                '\u{2588}' // █ (thumb)
            } else {
                '\u{2502}' // │ (track)
            };
            let cell_area = Rect {
                x: bar_x,
                y,
                width: 1,
                height: 1,
            };
            Paragraph::new(Line::from(Span::styled(
                ch.to_string(),
                Style::default().fg(Color::DarkGray),
            )))
            .render(cell_area, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Inline word-level diff helpers
// ---------------------------------------------------------------------------

/// Format a 10-char line-number gutter.
fn format_gutter(old_no: Option<u32>, new_no: Option<u32>) -> String {
    match (old_no, new_no) {
        (Some(o), Some(n)) => format!("{:>4} {:>4} ", o, n),
        (Some(o), None) => format!("{:>4}      ", o),
        (None, Some(n)) => format!("     {:>4} ", n),
        (None, None) => "          ".to_string(),
    }
}

/// Truncate a list of owned spans so the total character count ≤ `max_chars`.
fn truncate_spans_to_width(spans: Vec<Span<'static>>, max_chars: usize) -> Vec<Span<'static>> {
    let mut remaining = max_chars;
    let mut result = Vec::new();
    for span in spans {
        if remaining == 0 {
            break;
        }
        let char_count: usize = span.content.chars().count();
        if char_count <= remaining {
            remaining -= char_count;
            result.push(span);
        } else {
            let truncated: String = span.content.chars().take(remaining).collect();
            remaining = 0;
            result.push(Span::styled(truncated, span.style));
        }
    }
    result
}

/// Compute word-level inline diff spans for an adjacent (removed, added) line pair.
/// Returns `(old_spans, new_spans)` where changed words have a highlighted background.
fn build_inline_diff_spans(old: &str, new: &str) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    use similar::{ChangeTag, TextDiff};

    let diff = TextDiff::from_words(old, new);
    let mut old_spans: Vec<Span<'static>> = Vec::new();
    let mut new_spans: Vec<Span<'static>> = Vec::new();

    for change in diff.iter_all_changes() {
        let s: String = change.to_string();
        match change.tag() {
            ChangeTag::Equal => {
                old_spans.push(Span::styled(s.clone(), Style::default().fg(Color::Red)));
                new_spans.push(Span::styled(s, Style::default().fg(Color::Green)));
            }
            ChangeTag::Delete => {
                old_spans.push(Span::styled(
                    s,
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::Rgb(150, 30, 30)),
                ));
            }
            ChangeTag::Insert => {
                new_spans.push(Span::styled(
                    s,
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::Rgb(30, 130, 30)),
                ));
            }
        }
    }

    (old_spans, new_spans)
}

/// Highlight a line of source code using syntect, returning ratatui Spans.
/// Falls back to plain styling if the language is not recognised.
fn highlight_code_line(line: &str, path: &str, base_style: Style) -> Vec<Span<'static>> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let ss = &*SYNTAX_SET;
    let ts = &*THEME_SET;

    let syntax = if let Some(s) = ss.find_syntax_by_extension(ext) {
        s
    } else {
        return vec![Span::styled(line.to_string(), base_style)];
    };

    let theme = ts
        .themes
        .get("base16-ocean.dark")
        .or_else(|| ts.themes.values().next());

    let theme = match theme {
        Some(t) => t,
        None => return vec![Span::styled(line.to_string(), base_style)],
    };

    let mut h = HighlightLines::new(syntax, theme);
    match h.highlight_line(line, ss) {
        Ok(ranges) => {
            let mut result = Vec::new();
            for (style, text) in ranges {
                if text.is_empty() {
                    continue;
                }
                // Blend syntect foreground with the diff color (added=green, removed=red)
                let fg = style.foreground;
                // Only apply syntect color when it's not a "default" near-white color
                let is_default = fg.r > 200 && fg.g > 200 && fg.b > 200;
                let color = if is_default {
                    // Use the diff marker color (passed in base_style)
                    base_style.fg.unwrap_or(Color::White)
                } else {
                    Color::Rgb(fg.r, fg.g, fg.b)
                };
                result.push(Span::styled(text.to_string(), Style::default().fg(color)));
            }
            if result.is_empty() {
                vec![Span::styled(line.to_string(), base_style)]
            } else {
                result
            }
        }
        Err(_) => vec![Span::styled(line.to_string(), base_style)],
    }
}

fn build_diff_lines(file: &FileDiffStats, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    // Gutter = 10 chars ("dddd dddd "), prefix marker = 3 chars ("+  " etc.)
    let gutter_width: usize = 10;
    let prefix_width: usize = 3;
    let avail = (width as usize).saturating_sub(gutter_width + prefix_width);

    for hunk in &file.hunks {
        let hunk_lines = &hunk.lines;
        let mut i = 0;
        while i < hunk_lines.len() {
            let diff_line = &hunk_lines[i];

            // Detect adjacent Removed → Added pair for inline word-level diff
            if diff_line.kind == DiffLineKind::Removed {
                if let Some(next_line) = hunk_lines.get(i + 1) {
                    if next_line.kind == DiffLineKind::Added {
                        let (old_spans, new_spans) =
                            build_inline_diff_spans(&diff_line.content, &next_line.content);

                        let mut removed_row = vec![
                            Span::styled(
                                format_gutter(diff_line.old_line_no, None),
                                Style::default().fg(Color::DarkGray),
                            ),
                            Span::styled("-  ", Style::default().fg(Color::Red)),
                        ];
                        removed_row.extend(truncate_spans_to_width(old_spans, avail));
                        lines.push(Line::from(removed_row));

                        let mut added_row = vec![
                            Span::styled(
                                format_gutter(None, next_line.new_line_no),
                                Style::default().fg(Color::DarkGray),
                            ),
                            Span::styled("+  ", Style::default().fg(Color::Green)),
                        ];
                        added_row.extend(truncate_spans_to_width(new_spans, avail));
                        lines.push(Line::from(added_row));

                        i += 2;
                        continue;
                    }
                }
            }

            // Standard single-line rendering
            let (marker, content_style) = match diff_line.kind {
                DiffLineKind::Header => (
                    Span::styled("@@ ", Style::default().fg(Color::Cyan)),
                    Style::default().fg(Color::Cyan),
                ),
                DiffLineKind::Added => (
                    Span::styled("+  ", Style::default().fg(Color::Green)),
                    Style::default().fg(Color::Green),
                ),
                DiffLineKind::Removed => (
                    Span::styled("-  ", Style::default().fg(Color::Red)),
                    Style::default().fg(Color::Red),
                ),
                DiffLineKind::Context => (
                    Span::styled("   ", Style::default().fg(Color::DarkGray)),
                    Style::default().fg(Color::White),
                ),
            };

            let ln_str = format_gutter(diff_line.old_line_no, diff_line.new_line_no);
            let content: String = diff_line.content.chars().take(avail).collect();

            let mut row = vec![
                Span::styled(ln_str, Style::default().fg(Color::DarkGray)),
                marker,
            ];

            // Apply syntax highlighting for code lines (not headers)
            if diff_line.kind == DiffLineKind::Header {
                row.push(Span::styled(content, content_style));
            } else {
                let highlighted = highlight_code_line(&content, &file.path, content_style);
                row.extend(highlighted);
            }

            lines.push(Line::from(row));

            i += 1;
        }
    }

    lines
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file(path: &str, added: u32, removed: u32, is_new: bool) -> FileDiffStats {
        FileDiffStats {
            path: path.to_string(),
            added,
            removed,
            binary: false,
            is_new_file: is_new,
            is_deleted_file: false,
            patch_id: None,
            turn_index: None,
            tool_names: Vec::new(),
            hunks: Vec::new(),
        }
    }

    #[test]
    fn parse_unified_diff_new_file_flag() {
        let text = "diff --git a/new.rs b/new.rs\n\
                    new file mode 100644\n\
                    index 0000000..1234567\n\
                    --- /dev/null\n\
                    +++ b/new.rs\n\
                    @@ -0,0 +1,2 @@\n\
                    +fn foo() {}\n\
                    +fn bar() {}\n";
        let files = parse_unified_diff(text);
        assert_eq!(files.len(), 1);
        assert!(
            files[0].is_new_file,
            "new file mode header should set is_new_file"
        );
        assert_eq!(files[0].added, 2);
    }

    #[test]
    fn parse_unified_diff_existing_file_not_new() {
        let text = "diff --git a/lib.rs b/lib.rs\n\
                    index 1111111..2222222 100644\n\
                    --- a/lib.rs\n\
                    +++ b/lib.rs\n\
                    @@ -1,1 +1,1 @@\n\
                    -old line\n\
                    +new line\n";
        let files = parse_unified_diff(text);
        assert_eq!(files.len(), 1);
        assert!(!files[0].is_new_file);
    }

    #[test]
    fn parse_unified_diff_deleted_file_flag() {
        let text = "diff --git a/old.rs b/old.rs\n\
                    deleted file mode 100644\n\
                    index 1234567..0000000\n\
                    --- a/old.rs\n\
                    +++ /dev/null\n\
                    @@ -1,1 +0,0 @@\n\
                    -fn old() {}\n";
        let files = parse_unified_diff(text);
        assert_eq!(files.len(), 1);
        assert!(files[0].is_deleted_file);
        assert_eq!(files[0].removed, 1);
    }

    #[test]
    fn build_file_diff_from_snapshots_new_file_when_before_empty() {
        let file = build_file_diff_from_snapshots("src/new.rs".to_string(), "", "fn hello() {}\n");
        assert!(
            file.is_new_file,
            "empty before_text should mark file as new"
        );
        assert_eq!(file.added, 1);
        assert_eq!(file.removed, 0);
    }

    #[test]
    fn build_file_diff_from_snapshots_not_new_when_before_present() {
        let file = build_file_diff_from_snapshots(
            "src/lib.rs".to_string(),
            "fn old() {}\n",
            "fn new() {}\n",
        );
        assert!(!file.is_new_file);
    }

    #[test]
    fn build_file_diff_from_snapshots_deleted_when_after_empty() {
        let file = build_file_diff_from_snapshots("src/old.rs".to_string(), "fn old() {}\n", "");
        assert!(file.is_deleted_file);
        assert_eq!(file.removed, 1);
    }

    #[test]
    fn build_turn_diff_attaches_patch_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src").join("lib.rs");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut history = FileHistory::new();
        history.record_modification(path, b"fn old() {}\n", b"fn new() {}\n", 7, "FileEdit");

        let files = build_turn_diff(&history, 7, dir.path());
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].turn_index, Some(7));
        assert!(files[0]
            .patch_id
            .as_deref()
            .is_some_and(|id| id.starts_with("turn-7-")));
        assert_eq!(files[0].tool_names, vec!["FileEdit".to_string()]);
    }

    #[test]
    fn build_turn_diff_uses_existence_state_for_new_file_flag() {
        let dir = tempfile::tempdir().unwrap();
        let created = dir.path().join("created.rs");
        let deleted = dir.path().join("deleted.rs");
        let existing_empty = dir.path().join("existing_empty.rs");

        let mut history = FileHistory::new();
        history.record_modification_with_existence(
            created,
            b"",
            b"fn created() {}\n",
            (false, true),
            8,
            "FileWrite",
        );
        history.record_modification_with_existence(
            deleted,
            b"",
            b"",
            (true, false),
            8,
            "ApplyPatch",
        );
        history.record_modification(existing_empty, b"", b"fn filled() {}\n", 8, "FileWrite");

        let files = build_turn_diff(&history, 8, dir.path());
        assert_eq!(files.len(), 3);
        let created = files.iter().find(|file| file.path == "created.rs").unwrap();
        assert!(created.is_new_file);
        let deleted = files.iter().find(|file| file.path == "deleted.rs").unwrap();
        assert!(deleted.is_deleted_file);
        let existing = files
            .iter()
            .find(|file| file.path == "existing_empty.rs")
            .unwrap();
        assert!(!existing.is_new_file);
    }

    #[test]
    fn build_inline_diff_spans_equal_content() {
        let (old, new) = build_inline_diff_spans("hello world", "hello world");
        // All spans should have no background (equal, not highlighted)
        for span in &old {
            assert!(
                span.style.bg.is_none(),
                "equal spans should have no bg highlight"
            );
        }
        for span in &new {
            assert!(
                span.style.bg.is_none(),
                "equal spans should have no bg highlight"
            );
        }
        // Combined text should contain the key words
        let old_text: String = old.iter().map(|s| s.content.as_ref()).collect::<String>();
        let new_text: String = new.iter().map(|s| s.content.as_ref()).collect::<String>();
        assert!(
            old_text.contains("hello"),
            "old text should contain 'hello'"
        );
        assert!(
            new_text.contains("world"),
            "new text should contain 'world'"
        );
    }

    #[test]
    fn build_inline_diff_spans_highlights_changed_word() {
        let (old_spans, new_spans) = build_inline_diff_spans("hello world", "hello earth");
        // "world" should be highlighted (deleted), "earth" should be highlighted (inserted)
        let has_highlighted_old = old_spans
            .iter()
            .any(|s| s.content.contains("world") && s.style.bg.is_some());
        let has_highlighted_new = new_spans
            .iter()
            .any(|s| s.content.contains("earth") && s.style.bg.is_some());
        assert!(has_highlighted_old, "deleted word should have bg highlight");
        assert!(
            has_highlighted_new,
            "inserted word should have bg highlight"
        );
    }

    #[test]
    fn build_diff_lines_inline_diff_for_adjacent_pair() {
        let file = FileDiffStats {
            path: "test.rs".to_string(),
            added: 1,
            removed: 1,
            binary: false,
            is_new_file: false,
            is_deleted_file: false,
            patch_id: None,
            turn_index: None,
            tool_names: Vec::new(),
            hunks: vec![DiffHunk {
                old_range: (1, 1),
                new_range: (1, 1),
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        content: "let x = 1;".to_string(),
                        old_line_no: Some(1),
                        new_line_no: None,
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        content: "let x = 2;".to_string(),
                        old_line_no: None,
                        new_line_no: Some(1),
                    },
                ],
            }],
        };
        let lines = build_diff_lines(&file, 80);
        // Should produce 2 lines (one removed, one added)
        assert_eq!(
            lines.len(),
            2,
            "adjacent removed+added should produce 2 lines"
        );
        // Each line should have multiple spans (gutter + marker + content spans)
        assert!(lines[0].spans.len() >= 3);
        assert!(lines[1].spans.len() >= 3);
    }

    #[test]
    fn format_gutter_both_line_numbers() {
        let g = format_gutter(Some(10), Some(20));
        assert_eq!(g.len(), 10, "gutter should always be 10 chars");
        assert!(g.contains("10"));
        assert!(g.contains("20"));
    }

    #[test]
    fn format_gutter_old_only() {
        let g = format_gutter(Some(5), None);
        assert_eq!(g.len(), 10);
        assert!(g.contains("5"));
    }

    #[test]
    fn format_gutter_new_only() {
        let g = format_gutter(None, Some(99));
        assert_eq!(g.len(), 10);
        assert!(g.contains("99"));
    }

    #[test]
    fn truncate_spans_to_width_exact() {
        let spans = vec![Span::raw("hello"), Span::raw(" world")];
        let result = truncate_spans_to_width(spans, 11);
        let text: String = result.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn truncate_spans_to_width_cuts_mid_span() {
        let spans = vec![Span::raw("abcdefghij")];
        let result = truncate_spans_to_width(spans, 5);
        let text: String = result.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "abcde");
    }

    #[test]
    fn truncate_path_tail_is_char_boundary_safe() {
        assert_eq!(truncate_path_tail("src/über/long_file.rs", 8), "…file.rs");
        assert_eq!(truncate_path_tail("abc", 0), "");
        assert_eq!(truncate_path_tail("abc", 1), "…");
    }

    #[test]
    fn file_stats_binary_renders_badge() {
        // Verify the binary badge logic in render_file_list: binary=true → "[binary]"
        let file = FileDiffStats {
            path: "image.png".to_string(),
            added: 0,
            removed: 0,
            binary: true,
            is_new_file: false,
            is_deleted_file: false,
            patch_id: None,
            turn_index: None,
            tool_names: Vec::new(),
            hunks: Vec::new(),
        };
        let (stats, _color) = if file.binary {
            ("[binary]".to_string(), ratatui::style::Color::DarkGray)
        } else if file.is_new_file {
            (
                format!("[new] +{}", file.added),
                ratatui::style::Color::Yellow,
            )
        } else if file.is_deleted_file {
            (
                format!("[deleted] -{}", file.removed),
                ratatui::style::Color::Red,
            )
        } else {
            (
                format!("+{} -{}", file.added, file.removed),
                ratatui::style::Color::DarkGray,
            )
        };
        assert_eq!(stats, "[binary]");
    }

    #[test]
    fn file_stats_new_file_renders_badge() {
        let file = make_file("src/new.rs", 42, 0, true);
        let (stats, color) = if file.binary {
            ("[binary]".to_string(), ratatui::style::Color::DarkGray)
        } else if file.is_new_file {
            (
                format!("[new] +{}", file.added),
                ratatui::style::Color::Yellow,
            )
        } else if file.is_deleted_file {
            (
                format!("[deleted] -{}", file.removed),
                ratatui::style::Color::Red,
            )
        } else {
            (
                format!("+{} -{}", file.added, file.removed),
                ratatui::style::Color::DarkGray,
            )
        };
        assert_eq!(stats, "[new] +42");
        assert_eq!(color, ratatui::style::Color::Yellow);
    }

    #[test]
    fn file_stats_deleted_file_renders_badge() {
        let mut file = make_file("src/old.rs", 0, 7, false);
        file.is_deleted_file = true;
        let (stats, color) = if file.binary {
            ("[binary]".to_string(), ratatui::style::Color::DarkGray)
        } else if file.is_new_file {
            (
                format!("[new] +{}", file.added),
                ratatui::style::Color::Yellow,
            )
        } else if file.is_deleted_file {
            (
                format!("[deleted] -{}", file.removed),
                ratatui::style::Color::Red,
            )
        } else {
            (
                format!("+{} -{}", file.added, file.removed),
                ratatui::style::Color::DarkGray,
            )
        };
        assert_eq!(stats, "[deleted] -7");
        assert_eq!(color, ratatui::style::Color::Red);
    }

    #[test]
    fn diff_viewer_collapse_initializes_false() {
        let mut state = DiffViewerState::new();
        // Directly set files to simulate reload
        state.files = vec![
            make_file("a.rs", 1, 0, false),
            make_file("b.rs", 2, 1, false),
        ];
        state.collapsed = vec![false; state.files.len()];
        assert_eq!(state.collapsed.len(), 2);
        assert!(state.collapsed.iter().all(|&c| !c));
    }

    #[test]
    fn diff_viewer_toggle_collapse_selected() {
        let mut state = DiffViewerState::new();
        state.files = vec![
            make_file("a.rs", 1, 0, false),
            make_file("b.rs", 2, 1, false),
        ];
        state.collapsed = vec![false; 2];
        state.selected_file = 1;
        state.toggle_file_collapse();
        assert!(!state.collapsed[0], "file 0 should remain expanded");
        assert!(state.collapsed[1], "file 1 should now be collapsed");
        assert_eq!(state.detail_scroll, 0, "scroll resets on collapse");
    }

    #[test]
    fn diff_viewer_toggle_collapse_twice_restores() {
        let mut state = DiffViewerState::new();
        state.files = vec![make_file("a.rs", 1, 0, false)];
        state.collapsed = vec![false];
        state.toggle_file_collapse();
        assert!(state.collapsed[0]);
        state.toggle_file_collapse();
        assert!(!state.collapsed[0]);
    }

    #[test]
    fn diff_viewer_toggle_collapse_empty_files_no_panic() {
        let mut state = DiffViewerState::new();
        // No files — toggle should not panic
        state.toggle_file_collapse();
    }

    #[test]
    fn diff_viewer_set_turn_diff_resets_collapsed() {
        let mut state = DiffViewerState::new();
        state.diff_type = DiffType::TurnDiff;
        state.files = vec![make_file("x.rs", 1, 0, false)];
        state.collapsed = vec![true]; // manually set collapsed
        let new_files = vec![
            make_file("y.rs", 2, 0, false),
            make_file("z.rs", 3, 0, false),
        ];
        state.set_turn_diff(new_files);
        assert_eq!(
            state.collapsed.len(),
            2,
            "collapsed should match new file count"
        );
        assert!(
            state.collapsed.iter().all(|&c| !c),
            "new files start uncollapsed"
        );
    }

    #[test]
    fn diff_viewer_collapse_renders_without_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let mut state = DiffViewerState::new();
        state.open = true;
        state.files = vec![make_file("src/lib.rs", 5, 2, false)];
        state.collapsed = vec![true]; // collapsed
        terminal
            .draw(|frame| {
                let area = frame.area();
                render_diff_dialog(&mut state, area, frame.buffer_mut());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let content: String = buf
            .content()
            .iter()
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(content.contains("collapsed") || content.contains("Space"));
    }
}
