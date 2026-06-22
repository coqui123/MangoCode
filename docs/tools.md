# Tools

Tools are the actions the model can take — read a file, run a command, search the
web, spawn a sub-agent. They are implemented in the [`tools`](crates/tools.md)
crate. This page covers the tool framework, the permission model, and a catalog of
the built-in tools. For deeper API detail see [crates/tools.md](crates/tools.md).

---

## The `Tool` trait

Every tool implements the async `Tool` trait
([`crates/tools/src/lib.rs`](../src-rust/crates/tools/src/lib.rs)):

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn permission_level(&self) -> PermissionLevel;
    fn input_schema(&self) -> serde_json::Value;        // JSON Schema
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult;
    fn aliases(&self) -> Vec<String> { /* defaults */ }
    fn capabilities(&self, input: &Value) -> ToolCapabilities { /* defaults */ }
    fn to_definition(&self) -> ToolDefinition;
    fn to_runtime_spec(&self) -> ToolSpec;
}
```

- **`ToolResult`** — `{ content, is_error, metadata }`; constructed with
  `ToolResult::success(..)` / `::error(..)` and optional `.with_metadata(..)`.
- **`ToolCapabilities`** — describes whether the tool is mutating, parallel-safe,
  its affected paths / network targets, sandbox preference, cancellation support,
  and approval keys. The runtime uses this to batch parallel-safe reads while
  serializing conflicting mutations (`tool-runtime::plan_execution_batches`).

### `ToolContext`

Shared context passed to every `execute`:

- `working_dir`, `additional_dirs` and path validation (`is_path_allowed`,
  `resolve_and_validate_path`).
- `permission_mode` + `permission_handler` (the human-in-the-loop channel) and
  `check_permission(...)`.
- `cost_tracker`, `session_id`, `current_turn`, `non_interactive`.
- `file_history` (for `/rewind`) and per-session snapshot registry (for `/undo`).
- `mcp_manager` for MCP tool calls.
- `question_prompt_tx` for `AskUserQuestion` dialogs.
- `coordination_actor_id` / parent ids and `inject_coordination_inbox` for
  multi-agent coordination.
- `cancel_token` for cooperative cancellation.

Per-session shell state (cwd + exported env) and file snapshots are kept in
process-global registries keyed by `session_id`, so `Bash`/`PowerShell` persist
state across calls and writes can be undone.

---

## Permission model

`PermissionLevel` ranks what a tool can do:

| Level | Meaning |
| --- | --- |
| `None` | No permission needed (read-only / informational). |
| `ReadOnly` | Local read-only access. |
| `Network` | Outbound network access. |
| `Write` | Filesystem writes. |
| `Execute` | Arbitrary command execution. |
| `Dangerous` | Potentially sandbox-bypassing (browser, computer-use). |
| `Forbidden` | Unconditionally blocked. |

At dispatch the loop checks the level against the active `PermissionMode`
(`Default`, `AcceptEdits`, `BypassPermissions`, `Plan`). For shell tools it also
runs a **risk classifier** — `core::bash_classifier` (`Safe`→`Critical`) and
`core::ps_classifier` — which can auto-block destructive commands (e.g. `rm -rf /`,
fork bombs, `curl | bash`, disk wipes) or force a confirmation. Optionally an LLM
**permission critic** (`core::permission_critic`) evaluates intent with caching.
Secrets in tool I/O are redacted by `tools::redact`.

---

## Tool catalog

Tool availability is feature-gated (see
[getting-started.md](getting-started.md#build-features)). Names below are the
identifiers exposed to the model.

### Filesystem

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Read` | Read a file (line range; images, PDFs, notebooks). | ReadOnly |
| `Write` | Create/overwrite a file. | Write |
| `Edit` | Exact string replacement (with uniqueness check). | Write |
| `BatchEdit` | Apply multiple edits atomically. | Write |
| `ApplyPatch` | Apply a unified diff (supports dry-run). | Write |
| `NotebookEdit` | Edit Jupyter notebook cells. | Write |
| `ViewImage` | Inspect a local image (OCR fallback). | ReadOnly |

Writes go through atomic write helpers, record snapshots (undo) and file history
(rewind), and can preflight cross-session write conflicts via the coordination
store.

### Shell & process

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Bash` / `PtyBash` | Run shell commands with persistent cwd/env, timeout, background. | Execute |
| `PowerShell` | Windows PowerShell / pwsh with risk classification. | Execute |
| `REPL` | Persistent interpreter (bash / python / node). | Execute |
| `Sleep` | Async pause (non-blocking). | None |

Large output is reduced (`tools::output_reducers`) with raw logs saved under
`~/.mangocode/command_logs/` and readable via `ToolLogRead`.

### Search & navigation

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Glob` | Fast filename pattern matching. | ReadOnly |
| `Grep` | Ripgrep-style content search (filters, context, modes). | ReadOnly |
| `CodeSearch` | Identifier-aware / semantic code search. | ReadOnly |
| `ToolSearch` | Discover runtime-visible tools by name/keyword. | None |

### Web & research

| Tool | Purpose | Perm |
| --- | --- | --- |
| `WebFetch` | HTTP GET with HTML→text and optional rendered fallback. | Network |
| `WebSearch` | Web search with dedup and source preferences. | Network |
| `DocSearch` / `DocRead` | Documentation search / read with citations. | Network |
| `DeepRead` | Multi-stage research pipeline with verification. | Network |
| `RenderedFetch` | Script-aware extraction for sparse pages. | Network |
| `RemoteTrigger` | Cross-session event dispatch. | Network |

Fetched content is cached (`~/.mangocode/web_cache/`) and wrapped as untrusted
input; anti-bot heuristics (`bot_wall_sniff`, `browser_antibot`) help with
challenge pages.

### Browser & computer-use (feature-gated)

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Browser` | Persistent headless Chromium: navigate, screenshot, extract, click, type, evaluate, pass challenges. | Dangerous |
| `computer` | Cross-platform mouse/keyboard/screenshot (Anthropic computer-use spec). | Dangerous |

### Planning, tasks & coordination

| Tool | Purpose | Perm |
| --- | --- | --- |
| `TaskCreate` / `TaskGet` / `TaskUpdate` / `TaskList` / `TaskStop` / `TaskOutput` | Task store (durable to `~/.mangocode/tasks.json`). | None / mixed |
| `TodoWrite` | Per-session todo list. | None |
| `update_plan` | Update the session plan. | None |
| `EnterPlanMode` / `ExitPlanMode` | Plan-mode transitions. | None |
| `Brief` | Compress long context into a brief. | ReadOnly |
| `EnterWorktree` / `ExitWorktree` | Scoped work sessions. | None |
| `CronCreate` / `CronDelete` / `CronList` | Scheduled prompts (`~/.mangocode/scheduled_tasks.json`). | Write / None |
| `Agent` | Spawn a sub-agent (optional worktree isolation / background). | varies |
| `TeamCreate` / `TeamDelete` | Spawn/dissolve agent teams. | varies |
| `SendMessage` | Send a coordination message to an actor/broadcast. | None |
| `CoordinationStatus` / `CoordinationInbox` / `ClaimWork` / `ReleaseWork` | Multi-process coordination (SQLite-backed). | None |

### MCP & skills

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Skill` | Load & run a skill template (`skill="list"` to enumerate). | ReadOnly |
| `mcp__auth` | OAuth (PKCE) for an MCP server. | Network |
| `ListMcpResources` / `ReadMcpResource` | MCP resources. | Network |
| `mcp__<server>__<tool>` | Wrapped tools from connected MCP servers. | Execute |

### Code intelligence

| Tool | Purpose | Perm |
| --- | --- | --- |
| `LSP` | Language-server queries (hover, definition, references, symbols, diagnostics, call hierarchy, …). | ReadOnly |
| `ProjectGraph` | Build/query the project dependency graph (report/path/search/communities/export). | Write |
| `PrWatch` | Monitor a GitHub PR. | Network |

### Config, output & utility

| Tool | Purpose | Perm |
| --- | --- | --- |
| `Config` | Read/update session settings. | None |
| `StructuredOutput` | Emit JSON/Markdown/HTML reports. | Write |
| `ToolLogRead` | Read saved command logs. | ReadOnly |
| `AskUserQuestion` | Ask the user clarifying questions (single/multi-select + free text). | None |

---

## Registry & resolution

Tools are assembled by `tools::all_tools()` (feature-gated), then filtered by the
CLI allow/deny lists and the active agent's access level. Name resolution is
alias-aware (e.g. `shell`/`exec` → `Bash`, `read_file` → `Read`, auto camelCase →
snake_case), and `ToolSearch` plus fuzzy matching (`tool-runtime::fuzzy`) suggest
close names when a requested tool isn't found. MCP tools are wrapped with
`mcp__<server>__<tool>` names and bare aliases.
