# Crate: `tui`

Path: [`src-rust/crates/tui`](../../src-rust/crates/tui) ·
Core: [`src/lib.rs`](../../src-rust/crates/tui/src/lib.rs),
[`src/app.rs`](../../src-rust/crates/tui/src/app.rs)

`tui` is the interactive terminal UI, built on `ratatui` + `crossterm`. It owns the
application state, the event loop, the rendering pipeline, the prompt editor, and a
large catalog of overlays and dialogs.

---

## Terminal lifecycle (`lib.rs`)

- `setup_terminal()` / `setup_terminal_with_capabilities(...)` — enable raw mode,
  enter the alternate screen, and turn on mouse capture and bracketed paste, with a
  panic hook that restores the terminal if the main thread panics.
- `restore_terminal(...)` — reverse all of the above and show the cursor.
- `TerminalCapabilities::detect()` — adjust behavior for IDE-embedded terminals.
- `init_mascot(...)` — detect the terminal image protocol (Kitty → iTerm → Sixel →
  text) and encode the mango mascot for the greeting box.

A **hybrid** backend (`hybrid.rs`) can render the transcript into the main screen
buffer (preserving native scrollback) while reserving rows for the composer.

---

## App state & event loop (`app.rs`)

The `App` struct is the single source of UI state. It tracks, among much else:

- **Conversation** — messages, system annotations, streaming text, scroll offset,
  status message.
- **Tools & permissions** — active `ToolUseBlock`s, the in-flight
  `PermissionRequest`, and the `QuestionDialog` from `AskUserQuestion`.
- **Model & cost** — model name, token count, cost, `effort_level`, fast mode,
  agent mode/status.
- **Overlays** — one field per overlay/dialog (see catalog below), plus a
  `ModalOwner` indicating which overlay currently owns focus and a `FocusTarget`
  (`Input` vs `Transcript`).
- **Session & integration** — session id, bridge state, remote session URL,
  coordination unread count, file history, MCP manager, file-search index.
- **Voice** — recorder, recording flag, transcription event channel.

`handle_key_event(key)` is the central keystroke router (returns true on submit);
`handle_query_event(event)` applies streaming updates from the query loop;
`intercept_slash_command(name)` dispatches slash commands.

---

## Rendering (`render.rs`, `messages/`, `virtual_list.rs`)

- `render_app(frame, app)` draws, in order: header → transcript → overlays →
  prompt → footer.
- **Message rendering** (`messages/mod.rs`) dispatches per role/content block:
  assistant/user text, thinking blocks, tool-use calls and results, code blocks.
- **Markdown** (`messages/markdown.rs`, `markdown_enhanced.rs`) renders rich text,
  tables, and inline formatting into styled `ratatui` lines.
- **Virtual list** (`virtual_list.rs`) provides efficient, height-cached scrolling
  of the message pane with sticky-bottom auto-follow.
- **Themes & figures** (`theme_colors.rs`, `figures.rs`) define color palettes
  (including a deuteranopia-friendly theme) and Unicode glyphs.

### Inline images (`kitty_image.rs`, `rustle.rs`, `image_paste.rs`)

Image output supports **Kitty** (APC), **iTerm** (OSC 1337), and **Sixel**, with
an LRU cache of transcript image ids. SVGs (the mascot) are rasterized to PNG and
encoded per protocol. Clipboard image paste is supported.

---

## Prompt input (`prompt_input.rs`, `input.rs`)

A multi-line editor with:

- **Vim mode** — `Insert` / `Normal` / `Visual` / `VisualLine` / `VisualBlock` /
  `Command` / `Search`, operators (`d`/`c`/`y`/case), find (`f`/`F`/`t`/`T`),
  counts, marks, registers, and dot-repeat.
- **History** — up/down navigation and Ctrl+R search.
- **Paste handling** — burst detection turns large pastes into placeholder
  attachments (`PastedDocument`).
- **Typeahead** — slash-command, `@file`/`@folder`/`@symbol`/`@recent`, and command
  completion, backed by `file-search`.

`input.rs` parses slash commands (`is_slash_command`, `parse_slash_command`) and
decides which commands with args bypass the TUI overlay.

---

## Overlays & dialogs

The crate ships ~40 overlays/screens, each in its own module. Highlights:

| Module | Trigger | Purpose |
| --- | --- | --- |
| `overlays.rs` | F1 / Ctrl+R / Ctrl+P | Help, history search, global ripgrep search, rewind selector. |
| `dialogs.rs` | tool permission / questions | Permission dialogs (`Generic`/`Bash`/`FileRead`/`FileWrite`) and `AskUserQuestion` forms; MCP approval. |
| `model_picker.rs` | `/model` | Model + effort/fast picker. |
| `settings_screen.rs` | `/config` | Full-screen settings. |
| `theme_screen.rs` / `privacy_screen.rs` | `/theme`, `/privacy-settings` | Theme and privacy. |
| `stats_dialog.rs` | `/stats` | Token/cost analytics. |
| `mcp_view.rs` | `/mcp` | MCP server/tool browser. |
| `agents_view.rs` | `/agents` | Agent definitions and coordinator status. |
| `diff_viewer.rs` | `/diff`, `/changes` | Git/turn diff viewer with per-file hunks. |
| `session_browser.rs` / `session_branching.rs` | `/session`, Ctrl+B | Session management and branching. |
| `tasks_overlay.rs` | Ctrl+T | Task progress. |
| `export_dialog.rs` | `/export` | JSON/Markdown export. |
| `context_viz.rs` | `/context` | Context-window visualization. |
| `elicitation_dialog.rs` | MCP elicitation | Form inputs requested by MCP servers. |
| `device_auth_dialog.rs` / `key_input_dialog.rs` | login | OAuth device flow / masked API-key entry. |
| `onboarding_dialog.rs`, `feedback_survey.rs`, `bypass_permissions_dialog.rs`, `hooks_config_menu.rs`, `memory_file_selector.rs`, `invalid_config_dialog.rs` | various | Onboarding, feedback, bypass confirmation, hooks viewer, memory picker, config-error dialog. |
| `dialog_select.rs` | Ctrl+K, `/connect` | Reusable fuzzy selection widget (command palette, provider picker). |
| `notifications.rs`, `*_notice.rs`, `*_upsell.rs`, `plugin_views.rs`, `bridge_state.rs` | banners | Notifications, voice/memory notices, upsell banners, plugin hints, bridge status. |

---

## Voice (`voice_capture.rs`)

Captures microphone audio via `cpal`, encodes WAV, and transcribes through a
Whisper endpoint, emitting `VoiceEvent`s consumed by the app loop. Surfaced by
`/voice`.

---

## Slash command metadata (`slash_commands.rs`)

`SlashCommandSpec` (name, description, group) plus a static list of prompt slash
commands and an autocomplete name list used by the prompt typeahead. The
authoritative command implementations live in the [`commands`](commands.md) crate.
