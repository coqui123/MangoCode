# Crate: `query`

Path: [`src-rust/crates/query`](../../src-rust/crates/query) ·
Core: [`src/lib.rs`](../../src-rust/crates/query/src/lib.rs)

`query` is the agentic engine: it drives the turn loop, streams model responses,
dispatches tools, manages context, and hosts a set of supporting subsystems
(compaction, sub-agents, schedulers, memory, work-run tracking, plan search,
proactive monitoring, coordination).

---

## The turn loop

The public entry point is:

```rust
pub async fn run_query_loop(
    client: &mangocode_api::AnthropicClient,
    messages: &mut Vec<Message>,
    tools: &[Box<dyn Tool>],
    tool_ctx: &ToolContext,
    config: &QueryConfig,
    cost_tracker: Arc<CostTracker>,
    event_tx: Option<UnboundedSender<QueryEvent>>,
    cancel_token: CancellationToken,
    pending_messages: Option<&mut Vec<String>>,
    permission_prompt_tx: Option<UnboundedSender<PermissionPrompt>>,
) -> QueryOutcome
```

It sets up harness recording and a `WorkRun` tracker, optionally runs plan search,
then delegates to the inner loop. Each turn: drain queued input → build the request
(system prompt + tool definitions + messages) → stream the response → handle the
stop reason → dispatch tools → feed results back → maintain context (compaction,
budget). See [architecture.md](../architecture.md#the-turn-loop) for the diagram.

### `QueryOutcome`

| Variant | Meaning |
| --- | --- |
| `EndTurn { message, usage }` | Finished naturally and the completion gate passed. |
| `MaxTokens { partial_message, usage }` | Output limit reached. |
| `Cancelled` | Cancelled by user/system. |
| `Error(ClaudeError)` | Unrecoverable error. |
| `BudgetExceeded { cost_usd, limit_usd }` | Cost cap hit. |

### `QueryEvent`

Events streamed to the UI: `Stream` / `StreamWithParent` (model stream),
`ToolStart` / `ToolEnd`, `TurnComplete`, `Status`, `Error`, and `TokenWarning`.

### `QueryConfig`

The per-run configuration. Highlights (full list in source):

- **Model & sampling** — `model`, `max_tokens`, `max_turns`, `temperature`,
  `thinking_budget`, `effort_level`, `fallback_model`, `qwen_preserve_thinking`.
- **Prompt** — `system_prompt`, `append_system_prompt`, `output_style`,
  `working_directory`.
- **Budgets** — `tool_result_budget`, `max_budget_usd`.
- **Registries** — `provider_registry`, `model_registry`, `oauth_provider`.
- **Agent control** — `agent_name`, `agent_definition`, `agent_mode`,
  `completion_policy`, `verification_policy`, `reliability_profile`,
  `speed_profile`.
- **Injection** — `command_queue`, `skill_index`, `injected_skills`,
  `skill_qa_blocks`, `inject_coordination_inbox`.

Constructors: `QueryConfig::from_config(&Config)` and
`from_config_with_registry(&Config, &ModelRegistry)`.

---

## Context management

### Compaction (`compact.rs`)

Keeps the conversation within the model's context window:

- **Auto-compact** at ~90% of the window: summarize older message groups into a
  single `<compact-summary>` block, keep the most recent groups verbatim.
- **Micro-compact** proactively at ~75%.
- **Reactive / collapse / snip** strategies for read-heavy or oversized tails.
- A **circuit breaker** disables compaction after repeated failures.
- `TokenWarningState` (`Ok` / `Warning` ≥80% / `Critical` ≥95%) drives UI warnings.

The summary is produced by a single non-agentic API call so compaction can't
recurse. `context_window_for_model(model)` resolves the window size.

### Context analyzer (`context_analyzer.rs`)

Powers `/ctx-viz`: breaks token usage into categories (system prompt, tool
definitions, history, tool results, attachments) and suggests a compaction
strategy.

### Execution scratchpad (`execution_scratchpad.rs`)

Injects a compact `[SCRATCHPAD]` (current plan / last tool result / next action)
into the dynamic prompt each turn to reduce goal drift. On by default
(`FLAG_EXECUTION_SCRATCHPAD`).

---

## Work-run tracking (`work_run.rs`)

A `WorkRun` tracks the lifecycle of a run: phase, changed files (with versions and
line ranges), source paths and evidence, verification candidates/attempts, and a
`CompletionReadiness` (ready flag, blockers, ungrounded changes, unresolved
risks). The loop consults this at `end_turn` to apply **completion** and
**verification** gates per `completion_policy` / `verification_policy`, optionally
continuing the run with a gate prompt instead of finishing.

---

## Sub-agents & coordination

- **`agent_tool.rs`** — backs the `Agent` (and team) tools: spawn a specialized
  sub-agent reusing the same loop, optionally in an isolated git worktree or in the
  background (`poll_background_agent(id)` to collect results).
- **`coordinator.rs`** — `AgentMode` (`Coordinator` / `Worker` / `WorktreeWorker` /
  `Normal`) and tool filtering. The coordinator is limited to orchestration tools
  and delegates execution to workers; helpers build the coordinator system prompt
  and worker context.
- **`command_queue.rs`** — a priority queue (`Interrupt`/`High`/`Normal`/`Low`)
  shared with the TUI; drained into injectable messages at the start of each turn.

---

## Memory subsystems

- **`session_memory.rs`** — extract durable facts (categories: user preference,
  project fact, code pattern, decision, constraint) from the conversation into
  topic files, gated on conversation depth.
- **`memory_loader.rs`** — load `MEMORY.md` and relevant topic files per query.
- **`auto_dream.rs`** — periodic background memory consolidation, gated on time +
  new-session count + a file lock.
- **`away_summary.rs`** — a short "while you were away" recap (cheap model) when you
  return to an idle session.

See [memory-skills-plugins.md](../memory-skills-plugins.md) for the full memory
architecture.

---

## Schedulers & background services

- **`cron_scheduler.rs`** — fires cron-scheduled prompts (from the `Cron*` tools) in
  background sub-query loops on minute boundaries.
- **`proactive.rs`** — an observe-only background agent that periodically gathers
  project state (git changes, LSP diagnostics, CI/PR status, pending tasks) and asks
  whether anything is worth surfacing. Restricted to read tools. Feature
  `proactive`.
- **`copilot_server.rs`** — auto-starts the local Copilot-Pirate proxy and refreshes
  its tokens when the `lmstudio` slot points at it.
- **`ollama.rs`** — ensures a local Ollama server is running before dispatch.
- **`skill_prefetch.rs`** — background scan that builds an in-memory `SkillIndex`
  (`SharedSkillIndex`) of bundled + discovered skills for fast lookup.

---

## Plan search (opt-in, expensive)

- **`plan_search.rs`** — Monte Carlo Tree Search over candidate plans, scored by a
  weighted combination of tests passing, clippy findings introduced, and a
  self-review critic (`combine_signals`). Off by default (`FLAG_PLAN_SEARCH`).
- **`plan_search_rollout.rs`** — realizes candidate plans in throwaway git worktrees
  to score them (agent or chat realization modes); guarded against re-entrancy.
- **`clippy_baseline.rs` / `clippy_diff.rs`** — compute clippy findings introduced
  relative to session start by reverting changed files behind a `RevertGuard`.

---

## Feature flags

The crate's tool features mirror the per-tool gates fanned out from `cli`
(`tool-bash`, `tool-read`, `tool-agent`, …) plus `proactive`. See the crate's
`Cargo.toml` and [getting-started.md](../getting-started.md#build-features).
