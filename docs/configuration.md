# Configuration

MangoCode's runtime behavior is driven by a merged `Config` (defined in
[`crates/core/src/lib.rs`](../src-rust/crates/core/src/lib.rs)), assembled from —
in increasing precedence — built-in defaults, `~/.mangocode/settings.json`,
project-local settings, enterprise/remote managed settings, environment variables,
and CLI flags.

---

## settings.json

The user settings file lives at `~/.mangocode/settings.json`. Project-local
overrides live at `.mangocode/settings.local.json` (keyed per git remote). On
startup, idempotent migrations (`core::migrations`) upgrade older settings to the
current schema (model alias renames, permission-mode moves, etc.).

Selected `Config` fields (see the source for the exhaustive list):

### Model & provider

| Field | Meaning |
| --- | --- |
| `model` | Default model (bare name or `provider/model`). |
| `provider` | Default provider id (defaults to `lmstudio`). |
| `effort` | Effort level (`low`/`medium`/`high`/`max`). |
| `max_tokens` | Default max response tokens. |
| `provider_configs` | Per-provider `ProviderConfig` (api key/base, enabled, model allow/deny lists, options). |

### Permissions & agent control

| Field | Meaning |
| --- | --- |
| `permission_mode` | `Default` / `AcceptEdits` / `BypassPermissions` / `Plan`. |
| `approvals_reviewer` | `User` (manual) or `AutoReview` (guardian sub-agent). |
| `critic_mode`, `critic_model` | Enable the LLM permission critic and the model it uses. |
| `agent_completion_policy` | `Enforce` / `Warn` / `Off`. |
| `verification_policy` | `Auto` / `Ask` / `Off`. |
| `agent_reliability_profile` | `Standard` / `Strict`. |
| `agent_speed_profile` | `Balanced` / `FastSafe`. |
| `allowed_tools`, `disallowed_tools` | Tool allow/deny lists. |

### UI & behavior

| Field | Meaning |
| --- | --- |
| `theme` | UI theme. |
| `output_style` | Output style name (see `core::output_styles`). |
| `auto_compact`, `compact_threshold` | Automatic compaction toggle and trigger fraction. |
| `output_format` | `Text` / `Json` / `StreamJson` (headless). |
| `prevent_idle_sleep` | Keep the OS awake during a turn (`sleep-inhibitor`). |
| `dry_run` | Preview writes without applying them. |

### Context, files & memory

| Field | Meaning |
| --- | --- |
| `custom_system_prompt`, `append_system_prompt` | Replace / extend the system prompt. |
| `disable_claude_mds` | Don't load `AGENTS.md` context. |
| `project_dir`, `workspace_paths`, `additional_dirs` | Project root and additional accessible directories. |
| `memory` | `MemoryConfig` — `layered_retrieval`, `embedding_provider`, `embedding_model`. |
| `research`, `attachments`, `tool_output` | Nested config for web research, attachment routing, and tool-output reduction. |

### Extensions

| Field | Meaning |
| --- | --- |
| `mcp_servers` | MCP server definitions (`McpServerConfig`). |
| `enable_all_mcp_servers` | Auto-enable all configured MCP servers. |
| `lsp_servers` | Language-server definitions for the LSP tool. |
| `formatter` | Per-filetype formatter commands. |
| `commands` | User-defined slash command templates. |
| `agents` | Named agent definitions (`AgentDefinition`). |
| `skills` | Skill discovery config (`paths`, `urls`). |
| `hooks` | Lifecycle hooks (see below). |
| `share_endpoint` | Session-share service URL. |

---

## Agents

`Config.agents` maps a name to an `AgentDefinition`:

| Field | Meaning |
| --- | --- |
| `description` | Shown in `@agent` autocomplete. |
| `model` | Per-agent model override. |
| `temperature` | Per-agent temperature. |
| `prompt` | System prompt prefix for the agent. |
| `access` | `full` / `read-only` / `search-only` — restricts the tool set. |
| `visible` | Whether it appears in autocomplete. |
| `max_turns` | Per-agent turn cap. |
| `color` | Display color. |

`read-only` agents have mutating tools removed and mutating action variants pruned
from tool schemas; `search-only` agents keep only read-oriented network tools.
Select an agent at launch with `-A`/`--agent` or manage them with `/agents`.

---

## Hooks

Hooks are shell commands that fire on lifecycle events. They come from two
sources: `Config.hooks` (user settings) and plugin `hooks.json` files (see
[crates/plugins.md](crates/plugins.md)). A hook receives the event payload as JSON
on stdin.

`HookEntry` fields:

| Field | Meaning |
| --- | --- |
| `command` | Shell command to run. |
| `tool_filter` | For `PreToolUse`/`PostToolUse`: only run for this tool (glob-matched). |
| `blocking` | If true, a non-zero exit blocks the operation. |

Hook events (`HookEvent` / plugin `HookEventKind`) include: `SessionStart`,
`SessionEnd`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`PostToolUseFailure`, `PreCompact`, `PostCompact`, `Stop`, `PostModelTurn`,
`Notification`, plus plugin-only events such as `PermissionDenied`,
`SubagentStart`/`SubagentStop`, `TaskCreated`/`TaskCompleted`,
`WorktreeCreate`/`WorktreeRemove`, `FileChanged`, and `CwdChanged`.

`PreToolUse` and `UserPromptSubmit` hooks can **block** (deny the action or reject
the prompt); `PostToolUse` hooks are informational. Hooks run with a short timeout
so a slow hook can't stall the loop. View configured hooks with `/hooks`.

---

## Feature flags

Runtime feature flags (`core::feature_flags`, persisted to
`~/.mangocode/flags.json`) toggle experimental behavior without recompiling. They
can be overridden by environment variables of the form
`MANGOCODE_FLAG_<UPPERCASE_NAME>=1|0|true|false|on|off`. Manage them with `/flags`.

| Flag | Default | Effect |
| --- | --- | --- |
| `proactive` | — | Background proactive-observation agent. |
| `llm_compaction` | on | LLM-assisted context compaction. |
| `cached_microcompact` | on | Cached micro-compaction. |
| `execution_scratchpad` | on | Per-turn plan/last-tool/next-action scratchpad injection. |
| `prompt_caching` | — | Provider prompt caching. |
| `auto_lsp` | — | Automatic LSP diagnostics. |
| `hierarchical_memory` | — | Hierarchical memory retrieval. |
| `preserve_thinking` | off | Preserve thinking blocks across turns. |
| `self_review` | off | Post-run self-critique. |
| `plan_search` | off | MCTS plan search (expensive). |
| `critic_permissions` | — | LLM permission critic. |

A separate, simpler mechanism — **feature gates** (`core::feature_gates`) — reads
`MANGOCODE_FEATURE_<NAME>` / `MANGOCODE_DYNAMIC_CONFIG_<NAME>` environment variables
with no persistence. `MANGOCODE_SIMPLE=1` (or `--bare`) enables **bare mode**.

---

## Environment variables

| Variable | Purpose |
| --- | --- |
| Provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, …) | Provider credentials. See [providers-and-models.md](providers-and-models.md). |
| `OLLAMA_HOST`, `LM_STUDIO_HOST`, `LLAMA_CPP_HOST` | Local model server endpoints. |
| `RUST_LOG` | Tracing filter. |
| `MANGOCODE_FLAG_<NAME>` | Override a runtime feature flag. |
| `MANGOCODE_FEATURE_<NAME>`, `MANGOCODE_DYNAMIC_CONFIG_<NAME>` | Feature gates / dynamic config. |
| `MANGOCODE_SIMPLE` | Bare mode. |
| `MANGOCODE_BRIDGE_URL`, `MANGOCODE_BRIDGE_TOKEN` | Remote-control bridge config (see [crates/bridge.md](crates/bridge.md)). |
| `MANGOCODE_EMBEDDINGS_PROVIDER`, `MANGOCODE_EMBEDDINGS_MODEL`, `MANGOCODE_EMBEDDINGS_COMMAND` | Layered-memory embeddings. |
| `MANGOCODE_DISABLE_AUTO_MEMORY`, `MANGOCODE_REMOTE` | Memory subsystem toggles. |
| `MANGOCODE_DISABLE_NONESSENTIAL_TRAFFIC` | Suppress background network (registry prefetch, etc.). |
| `AWS_REGION` / `AWS_DEFAULT_REGION`, AWS credentials | Bedrock. |

---

## Managed & synced settings

- **Enterprise/remote managed settings** (`core::remote_settings`) are fetched
  from the Anthropic API, cached at `~/.mangocode/remote-settings.json`, and
  refreshed periodically.
- **Settings sync** (`core::settings_sync`) can synchronize `settings.json` and
  `AGENTS.md` with claude.ai over OAuth, both user-scope and project-scope.
