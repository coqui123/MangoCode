# Architecture

This page explains how MangoCode is put together: the crate workspace, how the
crates depend on each other, the lifecycle of a turn through the agentic loop, the
core data types that flow between crates, and where state lives on disk.

All source lives under [`src-rust/`](../src-rust). The workspace manifest is
[`src-rust/Cargo.toml`](../src-rust/Cargo.toml).

---

## Workspace layout

MangoCode is a Cargo workspace (resolver v2, Rust 2021) with 15 crates. They fall
into four layers:

```
                         ┌──────────────────────────────┐
   binary / entrypoint   │            cli               │  → bin: `mangocode`
                         └──────────────────────────────┘
                                       │
        ┌──────────────┬──────────────┼───────────────┬───────────────┐
        ▼              ▼              ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │   tui   │   │ commands │   │  query   │   │   acp    │   │  bridge  │   feature layer
   └─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        │              │              │
        └──────────────┴──────┬───────┴───────────────┐
                              ▼                        ▼
                        ┌──────────┐            ┌──────────────┐
                        │  tools   │            │     mcp      │            services
                        └──────────┘            └──────────────┘
                              │                        │
        ┌──────────┬─────────┼──────────┬─────────────┤
        ▼          ▼         ▼          ▼              ▼
   ┌────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐
   │  api   │ │ plugins │ │tool-rt.  │ │file-search│ │turn-diff │           foundation
   └────────┘ └─────────┘ └──────────┘ └───────────┘ └──────────┘
        │                                                   
        └───────────────────────┬───────────────────────────┐
                                 ▼                            ▼
                          ┌──────────────┐           ┌──────────────────┐
                          │     core     │           │ sleep-inhibitor  │
                          └──────────────┘           └──────────────────┘
```

| Crate | Path | Responsibility |
| --- | --- | --- |
| `cli` (bin `mangocode`) | `crates/cli` | Argument parsing, startup wiring, run-mode dispatch, OAuth login flows. |
| `tui` | `crates/tui` | Interactive terminal UI: app state, event loop, rendering, overlays. |
| `commands` | `crates/commands` | Slash command framework + 100+ built-in commands; Chrome CDP client. |
| `query` | `crates/query` | The agentic turn loop and its subsystems (compaction, sub-agents, schedulers, memory, work-run tracking). |
| `acp` | `crates/acp` | Agent Client Protocol server (JSON-RPC over stdio) for editor/IDE hosts. |
| `bridge` | `crates/bridge` | Remote-control bridge to the web UI (long-poll protocol). |
| `tools` | `crates/tools` | `Tool` trait, `ToolContext`, permission model, and ~65 tool implementations. |
| `mcp` | `crates/mcp` | Model Context Protocol client (stdio + HTTP/SSE transports). |
| `api` | `crates/api` | Provider abstraction, streaming/SSE parsing, transformers, model registry. |
| `plugins` | `crates/plugins` | Plugin manifest parsing, loader, hook dispatch, registry. |
| `core` | `crates/core` | Shared types, config/settings, storage, auth/vault, memory, git, safety classifiers, utilities. |
| `tool-runtime` | `crates/tool-runtime` | Tool spec/capability types, execution batching, fuzzy name matching. |
| `file-search` | `crates/file-search` | File + code indexing, fuzzy match, TF-IDF code search. |
| `turn-diff` | `crates/turn-diff` | Per-turn unified diffs, patch bundles, rollback. |
| `sleep-inhibitor` | `crates/sleep-inhibitor` | Cross-platform OS sleep prevention during a turn. |

`core` is the foundation every other crate builds on; nothing depends on `cli`.

---

## Runtime startup

The binary entrypoint is [`crates/cli/src/main.rs`](../src-rust/crates/cli/src/main.rs).
Startup proceeds roughly as:

1. **Parse arguments** with `clap` into the `Cli` struct (full surface in
   [getting-started.md](getting-started.md#cli-reference)).
2. **Resolve the working directory** (default cwd or `--cwd`) and **load settings**
   from `~/.mangocode/settings.json`, applying migrations and merging CLI
   overrides and environment variables.
3. **Initialize tracing** (`tracing_subscriber`, filtered by `RUST_LOG`).
4. **Authenticate**: resolve credentials from the auth store / vault / env, or run
   an OAuth login flow if required.
5. **Build context**: assemble the system prompt (embedded base prompt + git/project
   context + memory + skills), discover MCP servers, and load plugins.
6. **Build tools**: assemble the feature-gated tool set, then filter by
   `--allowed-tools`/`--disallowed-tools` and the active agent's access level.
7. **Dispatch a run mode**:
   - **Interactive** — launch the `tui` app loop.
   - **Headless** (`-p`/`--print`) — run one prompt through the query loop and print
     the result (text / JSON / stream-JSON).
   - **ACP** — run the Agent Client Protocol server over stdio.
   - Optionally start the **bridge** for remote control (interactive only).

---

## The turn loop

The heart of MangoCode is the agentic loop in
[`crates/query/src/lib.rs`](../src-rust/crates/query/src/lib.rs). The public entry
is `run_query_loop(...)`, which sets up harness recording and a `WorkRun` tracker,
optionally runs plan search, then drives the inner loop. Each iteration ("turn"):

```
        ┌─────────────────────────────────────────────────────────────┐
        │  for each turn (until end_turn / max_turns / cancel / error) │
        └─────────────────────────────────────────────────────────────┘
                                   │
   1. Drain queued input ──────────┤  pending user messages + command queue
                                   │
   2. Build the request ──────────┤  system prompt (git ctx, memory, skills,
                                   │  execution scratchpad) + tool definitions
                                   │  (filtered by agent mode) + messages
                                   │
   3. Stream the response ────────┤  accumulate text / thinking / tool_use blocks;
                                   │  emit QueryEvent::Stream for the UI; track tokens
                                   │
   4. On stop_reason ─────────────┤
        • end_turn  → completion/verification gate → finish or continue
        • max_tokens → return MaxTokens
        • tool_use  → go to 5
                                   │
   5. Dispatch tools ─────────────┤  permission checks + bash/PowerShell risk
                                   │  classification; run tools; collect results
                                   │
   6. Feed results back ──────────┤  append tool_result blocks as a user message;
                                   │  update scratchpad + memory; fire post-tool hooks
                                   │
   7. Maintain context ───────────┤  token-warning checks; auto/micro compaction;
                                   │  budget guard
                                   │
        └──────────────────────────┘  loop
```

The loop terminates with a `QueryOutcome`:

| Variant | Meaning |
| --- | --- |
| `EndTurn` | Model finished and any completion gate passed. |
| `MaxTokens` | Output token limit reached; a partial message is returned. |
| `Cancelled` | User or system cancelled. |
| `Error` | Unrecoverable error (`ClaudeError`). |
| `BudgetExceeded` | Cumulative cost hit the configured USD cap. |

Behavior is governed by `QueryConfig` (model, token/turn limits, system prompt,
thinking budget, effort level, completion/verification policies, agent mode,
provider/model registries, and more). See
[crates/query.md](crates/query.md) for the full field list and every subsystem
(compaction, sub-agents, cron scheduler, memory, plan search, proactive monitor,
coordination).

---

## Core data model

The types that flow between crates are defined in
[`crates/core/src/lib.rs`](../src-rust/crates/core/src/lib.rs):

- **`Message`** — a `Role` (`User`/`Assistant`) plus `MessageContent` (text or a
  `Vec<ContentBlock>`), with an optional transcript `uuid` and `MessageCost`.
- **`ContentBlock`** — the rich content union: `Text`, `Image`, `ToolUse`,
  `ToolResult`, `Thinking`, `RedactedThinking`, `Document`, plus MangoCode-specific
  blocks like `UserLocalCommandOutput`, `UserCommand`, `SystemAPIError`,
  `CollapsedReadSearch`, and `TaskAssignment`.
- **`ToolDefinition`** — `name`, `description`, and a JSON-Schema `input_schema`.
- **`UsageInfo` / `MessageCost`** — token counts (input/output, cache create/read)
  and computed USD cost.
- **`ProviderId` / `ModelId`** — branded string newtypes (see
  [`provider_id.rs`](../src-rust/crates/core/src/provider_id.rs)) with constants
  for every known provider and helpers for `provider/model` parsing.
- **`Config`** — the merged runtime configuration (model, permissions, hooks,
  providers, agents, memory, research, attachments…); see
  [configuration.md](configuration.md).
- **`ClaudeError`** — the shared error enum (`Result<T> = Result<T, ClaudeError>`),
  with `is_retryable()` and `is_context_limit()` helpers.

The `api` crate additionally defines **provider-agnostic** wire types
(`ProviderRequest`, `ProviderResponse`, `StreamEvent`, `StopReason`) and the
Anthropic-native client (`AnthropicClient`, `StreamAccumulator`). See
[crates/api.md](crates/api.md).

---

## On-disk layout

MangoCode keeps user state in two homes:

### `~/.mangocode/` — configuration & state

| Path | Contents |
| --- | --- |
| `settings.json` | User settings (merged into `Config`). |
| `settings.local.json` | Project-local settings (per git remote). |
| `AGENTS.md` | User-scope memory/instructions. |
| `auth.json` | Stored provider credentials (`AuthStore`). |
| `vault.enc` | Optional AES-256-GCM encrypted secret vault. |
| `flags.json` | Runtime feature flags (`FeatureFlags`). |
| `usage.json` | Cumulative usage ledger (cost/tokens). |
| `history.jsonl` | Prompt history (with externalized large pastes under `pastes/`). |
| `sessions.db` | SQLite store for harness events, checkpoints, goals, coordination. |
| `tasks.json`, `scheduled_tasks.json`, `todos/` | Task store, cron tasks, per-session todos. |
| `command_logs/`, `traces/`, `web_cache/` | Tool output logs, session traces, web fetch cache. |
| `skills/`, `commands/`, `plugins/`, `output-styles/`, `rules/` | User-defined extensions. |
| `mcp-tokens/` | Per-server MCP OAuth tokens. |

### `~/.claude/projects/<encoded-project>/` — transcripts

Session transcripts are append-only JSONL files
(`<session_id>.jsonl`). The project path is encoded by replacing `/ \ : space ~ _`
with `-`. See `session_storage.rs` and [crates/core.md](crates/core.md#session--transcript-storage).

Project-scoped extensions are also discovered from the working tree:
`.mangocode/` (skills, commands, plugins, settings, output-styles), `.agents/`,
and `AGENTS.md` / `.mangocode/AGENTS.md`.

---

## Cross-cutting systems

- **Permissions** — every tool declares a `PermissionLevel`; the loop checks it
  against the active `PermissionMode`, optionally consulting bash/PowerShell risk
  classifiers (`core::bash_classifier`, `core::ps_classifier`) or an LLM
  `permission_critic`. See [tools.md](tools.md#permission-model).
- **Memory** — `AGENTS.md` hierarchy (`core::claudemd`), file-based memory dirs
  (`core::memdir`), and an optional SQLite layered store with embeddings
  (`core::layered_memory`). See [memory-skills-plugins.md](memory-skills-plugins.md).
- **Hooks** — config- and plugin-defined shell hooks fire on lifecycle events
  (`PreToolUse`, `PostToolUse`, `SessionStart`, `UserPromptSubmit`, …). See
  [configuration.md](configuration.md#hooks) and [crates/plugins.md](crates/plugins.md).
- **Harness & checkpoints** — `core::harness` records turn events and git/file
  checkpoints to enable `/undo` and `/rewind`.
- **Feature gating** — most tools and several subsystems are behind Cargo features
  (see [getting-started.md](getting-started.md#build-features)) and/or runtime
  feature flags (`core::feature_flags`).
