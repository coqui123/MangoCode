# Crate: `tools`

Path: [`src-rust/crates/tools`](../../src-rust/crates/tools) ·
Core: [`src/lib.rs`](../../src-rust/crates/tools/src/lib.rs)

`tools` defines the tool framework — the `Tool` trait, `ToolContext`, the
permission model, the registry, and ~65 tool implementations. This page is the API
reference; the catalog and permission model are covered in the user guide at
[../tools.md](../tools.md).

---

## Framework

### `Tool` trait

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn permission_level(&self) -> PermissionLevel;
    fn input_schema(&self) -> serde_json::Value;
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult;
    fn aliases(&self) -> Vec<String> { /* default aliases */ }
    fn capabilities(&self, input: &Value) -> ToolCapabilities { /* defaults */ }
    fn to_definition(&self) -> ToolDefinition;
    fn to_runtime_spec(&self) -> ToolSpec;
}
```

- **`ToolResult`** — `{ content, is_error, metadata }`; `success(..)` / `error(..)`
  / `with_metadata(..)` / `to_envelope()`.
- **`PermissionLevel`** — `None` / `ReadOnly` / `Network` / `Write` / `Execute` /
  `Dangerous` / `Forbidden`.
- **`ToolCapabilities`** (from `tool-runtime`) — mutating / parallel-safe /
  affected paths / network targets / sandbox preference / cancellation / approval
  keys; drives parallel execution batching.

### `ToolContext`

Carries `working_dir` + `additional_dirs` (with `is_path_allowed`,
`resolve_and_validate_path`), `permission_mode` + `permission_handler` (with
`check_permission(...)`), `cost_tracker`, `session_id`, `current_turn`,
`non_interactive`, `file_history`, the per-session snapshot registry,
`mcp_manager`, `question_prompt_tx`, coordination ids, and a `cancel_token`.

### Session state

Process-global registries keyed by `session_id`:

- **`SHELL_STATE_REGISTRY`** — persists `ShellState` (cwd + exported env) across
  `Bash`/`PowerShell` calls (`session_shell_state`, `clear_session_shell_state`).
- **`SNAPSHOT_REGISTRY`** — per-tool-call content snapshots for `/undo`
  (`session_snapshot`).

### Registry & resolution

`all_tools()` builds the feature-gated set; `find_tool(name)` and `resolve_tool`
do alias-aware lookup; `build_registry_plan(tools)` produces a `ToolRegistryPlan`;
`filter_tools_by_name_config(...)` applies allow/deny lists;
`tool_supports_parallel(...)` checks capabilities. MCP tools are wrapped with
`McpToolWrapper` under `mcp__<server>__<tool>` names + bare aliases.

---

## Tool modules

Tools are organized one (or a few) per file. By family:

| Family | Modules |
| --- | --- |
| Filesystem | `file_read`, `file_write`, `file_edit`, `batch_edit`, `apply_patch`, `notebook_edit`, `view_image`, `fs_atomic` |
| Shell / process | `bash`, `pty_bash`, `powershell`, `repl_tool`, `sleep` |
| Search | `glob_tool`, `grep_tool`, `code_search_tool`, `tool_search` |
| Web / research | `web_fetch`, `web_search`, `research`, `bot_wall_sniff`, `remote_trigger` |
| Browser / computer | `browser_tool`, `browser_antibot/*`, `computer_use` |
| Planning / tasks | `tasks`, `todo_write`, `update_plan`, `enter_plan_mode`, `exit_plan_mode`, `brief`, `worktree`, `cron`, `goal_tool` |
| Agents / coordination | `agent_tool`, `team_tool`, `send_message`, `coordination` |
| MCP / skills | `mcp_auth_tool`, `mcp_resources`, `skill_tool`, `bundled_skills` |
| Code intelligence | `lsp_tool`, `project_graph`, `pr_watch` |
| Config / output / HITL | `config_tool`, `synthetic_output`, `output_reducers`, `formatter`, `ask_user` |

### Shared infrastructure

- **`circuit_breaker.rs`** — per-key failure tracking that opens after a threshold.
- **`redact.rs`** — regex-based secret redaction (API keys, tokens, JWTs, PEM keys).
- **`output_reducers.rs`** — large-output reduction with raw logs on disk
  (`~/.mangocode/command_logs/`), surfaced via `ToolLogRead`.
- **`humanize.rs`** — human-readable durations.
- **`edit_hints.rs`** — line-ending normalization and "not found" hints for edits.
- **`ansi.rs`** — ANSI parsing/stripping.
- **`coordination.rs`** — write-claim preflight against the coordination store.

See [../tools.md](../tools.md) for the full per-tool catalog (names, parameters,
permission levels) and the permission/risk-classification model.
