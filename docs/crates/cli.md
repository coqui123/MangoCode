# Crate: `cli` (binary `mangocode`)

Path: [`src-rust/crates/cli`](../../src-rust/crates/cli) ·
Entry: [`src/main.rs`](../../src-rust/crates/cli/src/main.rs)

The `cli` crate is the MangoCode binary. It parses arguments, loads configuration,
authenticates, builds the system prompt / tools / MCP / plugins, and dispatches
into a run mode (interactive TUI, headless, or ACP). It also implements the OAuth
login flows.

---

## Startup flow

`main()` runs on a `tokio` runtime and roughly:

1. Parses the `Cli` struct with `clap`.
2. Resolves the working directory and loads `settings.json` (with migrations and
   CLI/env overrides).
3. Initializes `tracing` (filtered by `RUST_LOG`).
4. Resolves credentials (auth store / vault / env) or runs an OAuth flow.
5. Builds the session system prompt and project/user context.
6. Builds the tool set, MCP connections, and plugin registry.
7. Dispatches the run mode and optionally starts the bridge.

Build metadata (`VERSION`, `GIT_COMMIT`, `BUILD_TIME`, distribution URLs) is
embedded at compile time by `build.rs`. On Windows, `build.rs` also rasterizes
`mangoMan.svg` into a multi-resolution `.ico` and embeds it as the executable icon.

---

## Argument surface

CLI arguments are defined by the `Cli` struct (derive `clap::Parser`) and several
`ValueEnum` types that map to core config enums:

| CLI enum | Maps to |
| --- | --- |
| `CliPermissionMode` | `PermissionMode` |
| `CliCompletionPolicy` | `AgentCompletionPolicy` |
| `CliVerificationPolicy` | `VerificationPolicy` |
| `CliReliabilityProfile` | `AgentReliabilityProfile` |
| `CliSpeedProfile` | `AgentSpeedProfile` |
| `CliOutputFormat` / `CliInputFormat` | `OutputFormat` / input format |

The full flag list is documented in
[getting-started.md](../getting-started.md#cli-reference). Notable behavior:

- `--effort` is parsed by `parse_effort_level_arg` into `core::effort::EffortLevel`
  and can be persisted to settings.
- `--dump-system-prompt` builds and prints the assembled prompt, then exits.
- `--list-models` lists models for the configured providers, then exits.

---

## System prompt assembly

`build_session_system_prompt(working_dir, config, append_if_missing)`:

- Loads the embedded base prompt (`system_prompt.txt`).
- Adds system context (git state, project metadata) and user context (`AGENTS.md`,
  user instructions) via the context builder.
- Applies `custom_system_prompt` / `append_system_prompt`.
- Inserts the `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` marker that the Anthropic
  provider uses to split the cacheable (static) prefix from the dynamic suffix for
  prompt caching.

---

## Tool wiring

- `build_tools_with_mcp(...)` assembles feature-gated tools plus MCP tools, then
  applies `--allowed-tools` / `--disallowed-tools`.
- `filter_tools_for_agent(...)` restricts tools for an agent's `access` level:
  `read-only` drops mutating tools and prunes mutating action variants from
  schemas; `search-only` keeps only read-oriented network tools. Restriction is
  implemented by `AgentRestrictedTool` wrapping the inner tool.

---

## Permission & hooks

- Selects an interactive or automatic permission handler based on config and
  flags.
- `run_user_prompt_submit_hooks(...)` runs `UserPromptSubmit` hooks before
  submission; a hook can block or rewrite the prompt (short timeout, then continue).

---

## OAuth flows

### Claude (`src/oauth_flow.rs`)

`run_oauth_login_flow(login_with_claude_ai)` runs a PKCE loopback flow:

1. Generate verifier + challenge, bind a localhost listener.
2. Open the authorize URL (Console or claude.ai).
3. Race the browser redirect against manual code paste (timeout).
4. Exchange the code for tokens; for the Console flow, create an API key.
5. Persist `OAuthTokens`; return a `LoginResult` (credential, bearer flag, tokens).

`refresh_oauth_token(...)` refreshes via the stored refresh token.

### OpenAI Codex (`src/codex_oauth_flow.rs`)

`run_oauth_flow(event_tx)` / `run_terminal_oauth_flow()` run the ChatGPT/Codex PKCE
flow on a fixed loopback port, extract the ChatGPT account id from the access-token
JWT, and return `CodexTokens`. The TUI variant emits `DeviceAuthEvent`s so the
device-auth dialog can show progress.

---

## Bridge integration

`resolve_bridge_config(...)` returns a `BridgeConfig` only when not headless, the
bridge is enabled (settings / env), and a session token is available. When active,
the TUI run path starts the bridge loop so the session can be remote-controlled.
See [bridge.md](bridge.md).

---

## Dependencies

`cli` depends on every feature crate (`api`, `acp`, `tools`, `query`, `tui`,
`commands`, `mcp`, `bridge`, `plugins`) plus `core` (with the `voice` feature). Its
feature flags fan out per-tool flags to all downstream crates so a tool is
consistently compiled in or out (see
[getting-started.md](../getting-started.md#build-features)).
