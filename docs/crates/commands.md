# Crate: `commands`

Path: [`src-rust/crates/commands`](../../src-rust/crates/commands) ·
Core: [`src/lib.rs`](../../src-rust/crates/commands/src/lib.rs),
[`src/named_commands.rs`](../../src-rust/crates/commands/src/named_commands.rs)

`commands` implements the slash command framework and all built-in commands. The
full user-facing catalog is in [slash-commands.md](../slash-commands.md); this page
documents the framework.

---

## The `SlashCommand` trait

```rust
#[async_trait]
pub trait SlashCommand: Send + Sync {
    fn name(&self) -> &str;
    fn aliases(&self) -> Vec<&str> { vec![] }
    fn description(&self) -> &str;
    fn help(&self) -> &str { self.description() }
    fn hidden(&self) -> bool { false }
    async fn execute(&self, args: &str, ctx: &mut CommandContext) -> CommandResult;
}
```

Each command is a struct implementing this trait. `name()` is the invocation
(without `/`); `aliases()` are alternatives; `hidden()` omits it from `/help`;
`execute()` parses the argument string and returns a `CommandResult`.

---

## `CommandContext`

Passed (mutably) to every command. Fields include the current `Config`, the
`cost_tracker`, optional `session_metrics`, the conversation `messages`, the
`effort_level`, `working_dir`, `session_id` / `session_title`, the optional
`remote_session_url`, the `mcp_manager`, and the `model_registry`.

---

## `CommandResult`

The return value tells the host what to do:

| Variant | Effect |
| --- | --- |
| `Message(String)` | Show a message to the user (not added to the model conversation). |
| `UserMessage(String)` | Inject a message as if the user typed it (goes to the model). |
| `ConfigChange(Config)` | Apply a new config silently. |
| `ConfigChangeMessage(Config, String)` | Apply config and show a confirmation. |
| `ClearConversation` | Reset the conversation. |
| `SetMessages(Vec<Message>)` | Replace the message list (e.g. `/rewind`). |
| `ImportSessionState { … }` | Bulk import config + messages + effort + cwd in one step. |
| `ResumeSession(ConversationSession)` | Load a saved session. |
| `SetWorkingDir(PathBuf, String)` | Switch the workspace. |
| `RenameSession(String)` | Update the session title. |
| `StartOAuthFlow(bool)` | Start an OAuth flow (claude.ai vs Console). |
| `ReloadAuthStore(String)` | Reload credentials from disk + vault. |
| `OpenRewindOverlay` / `OpenHooksOverlay` | Open a TUI overlay (with text fallback). |
| `Exit` | Quit. |
| `Silent` | No output. |
| `Error(String)` | Show an error. |

---

## Registry

- `all_commands() -> Vec<Box<dyn SlashCommand>>` returns every built-in command.
- `find_command(name)` resolves a command case-insensitively by name or alias.

Built-ins are split between `lib.rs` (the bulk) and `named_commands.rs`
(`/agents`, `/add-dir`, `/branch`, `/vault`, `/desktop`, `/mobile`, `/pr-comments`,
`/ultraplan`, …). User- and plugin-defined templates and skills extend the set at
runtime.

---

## Chrome DevTools client (`chrome_cdp.rs`)

Backs the `/chrome` command with a Chrome DevTools Protocol client over WebSocket.
It resolves the debugger endpoint (env vars `MANGOCODE_CDP_WS` / `MANGOCODE_CDP_URL`,
`/json/version`, or the `DevToolsActivePort` file), then exposes async operations:
`connect` / `disconnect`, `navigate`, `screenshot`, `click`, `fill` /
`fill_keystrokes`, `eval_js` / `eval_in_iframe`, `tabs_list` / `switch_tab` /
`new_tab`, `page_info`, `handle_js_dialog`, and `wait_network_idle`. It supports
both browser-level (flattened `Target.*`) and page-direct sessions.

---

## Dependencies & features

Depends on `core`, `api`, `tools`, `query`, `tui`, `mcp`, and `plugins`, plus
`tokio-tungstenite` + `futures` (Chrome CDP) and CLI utility crates. Tool feature
flags mirror the per-tool gates fanned out from `cli`.
