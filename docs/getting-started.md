# Getting Started

This page covers building MangoCode, the available run modes, the full CLI
argument surface, and how to authenticate a model provider.

---

## Prerequisites

- **Rust** (2021 edition toolchain) with `cargo`.
- A **C toolchain** (for bundled native dependencies such as `rusqlite` and TLS).
- An account/credentials for at least one model provider, **or** a local model
  server (Ollama / LM Studio) — the default provider is `lmstudio`.

Some optional features pull heavier dependencies:

- `browser` / `computer-use` — headless Chromium and input/screenshot libraries.
- Local embeddings (`fastembed`) for the layered memory store.

---

## Building

From the workspace root (`src-rust/`):

```bash
cd src-rust

# Debug build
cargo build

# Optimized build
cargo build --release

# Run directly
cargo run --release --             # interactive
cargo run --release -- -p "hello"  # headless
```

The binary is `mangocode` (`target/release/mangocode`). On Windows the build embeds
the mango mascot SVG as the executable icon.

### Build features

The `cli` crate ([`crates/cli/Cargo.toml`](../src-rust/crates/cli/Cargo.toml))
controls which tools and subsystems are compiled in. Default features are
`proactive` + `default-tools`.

| Feature | Effect |
| --- | --- |
| `default-tools` | The standard tool set (file, shell, search, web, tasks, agent, coordination, project-graph, …). |
| `default-tools-no-web-research` | Same, minus the web/research tools. |
| `full-tools` | Everything, plus `browser`, `computer-use`, and `rendered-fetch`. |
| `proactive` | The background proactive-observation agent. |
| `browser` / `computer-use` | Enable just the browser / computer-use tools. |
| `tool-<name>` | Fine-grained per-tool gates (e.g. `tool-bash`, `tool-read`, `tool-web-search`). |

Each `tool-<name>` flag fans out to the matching feature in `tools`, `query`,
`commands`, `tui`, and `acp`, so a tool is consistently present (or absent) across
the whole stack. Build a minimal binary by disabling default features and opting
into only the tools you need:

```bash
cargo build --release --no-default-features --features "tool-read,tool-edit,tool-bash,tool-grep,tool-glob"
```

---

## Run modes

| Mode | How to enter | Behavior |
| --- | --- | --- |
| **Interactive TUI** | default (no `-p`, a TTY) | Full terminal UI: streaming transcript, slash commands, overlays. |
| **Headless / print** | `-p` / `--print`, or piping a prompt | Runs the prompt through the query loop and prints output, then exits. Output shape set by `--output-format`. |
| **ACP server** | invoked by an ACP host over stdio | JSON-RPC 2.0 Agent Client Protocol server (see [crates/acp.md](crates/acp.md)). |
| **Remote-controlled** | bridge enabled (interactive) | A local session driven from the web UI (see [crates/bridge.md](crates/bridge.md)). |

Headless example with structured output:

```bash
mangocode -p "summarize the architecture" --output-format json --model sonnet
```

---

## Authentication

Credentials are resolved from (in order) the auth store (`~/.mangocode/auth.json`),
the encrypted vault (`~/.mangocode/vault.enc`, if unlocked), and provider
environment variables. You can authenticate by:

- **API key** — set the provider's env var (e.g. `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GOOGLE_API_KEY`) or pass `--api-key`, or use `/login` / `/connect`
  in the TUI.
- **OAuth** — for Claude (Pro/Max via claude.ai or Console) and OpenAI Codex
  (ChatGPT). The TUI offers `/login` and `/connect`; the CLI runs a local PKCE
  loopback flow. See [`crates/cli/src/oauth_flow.rs`](../src-rust/crates/cli/src/oauth_flow.rs)
  and [`codex_oauth_flow.rs`](../src-rust/crates/cli/src/codex_oauth_flow.rs).
- **Vault** — `--vault-passphrase` / `--vault-prompt` to unlock encrypted secrets
  for the session, or the `/vault` command.

The complete provider list and their environment variables are in
[providers-and-models.md](providers-and-models.md).

---

## CLI reference

Arguments are parsed by the `Cli` struct in
[`crates/cli/src/main.rs`](../src-rust/crates/cli/src/main.rs). The most useful
flags are grouped below.

### Prompt & mode

| Flag | Description |
| --- | --- |
| `[PROMPT]` | Positional initial prompt; presence implies headless intent. |
| `-p`, `--print` | Headless mode: run once and exit. |
| `-c`, `--continue` | Resume the most recent session. |
| `--resume <ID>` | Resume a specific session by id. |
| `--session-id <ID>` | Tag a headless run with a session id. |
| `--prefill <TEXT>` | Pre-fill the first assistant response. |

### Model & provider

| Flag | Description |
| --- | --- |
| `-m`, `--model <MODEL>` | Model override (bare name or `provider/model`). |
| `--provider <ID>` | Provider override (e.g. `anthropic`, `openai`, `ollama`, `lmstudio`). |
| `--api-base <URL>` | Override the provider API base URL. |
| `--api-key <KEY>` | Override the API key. |
| `--fallback-model <MODEL>` | Model to fall back to on overload. |
| `--betas <CSV>` | Extra beta headers. |
| `--list-models` | List available models and exit. |
| `--workload <TAG>` | Billing workload tag. |

### Reasoning & limits

| Flag | Description |
| --- | --- |
| `--effort <LEVEL>` | `low` / `medium` (alias `normal`) / `high` / `max`. |
| `--thinking <TOKENS>` | Extended-thinking token budget. |
| `--max-tokens <N>` | Max response tokens. |
| `--max-turns <N>` | Max agentic turns. |
| `--max-budget-usd <USD>` | Hard spend cap for the run. |
| `--qwen-preserve-thinking` | Preserve Qwen reasoning blocks. |

### Agent behavior

| Flag | Description |
| --- | --- |
| `-A`, `--agent <NAME>` | Named agent (e.g. build, plan, explore, review, test, docs, debug, refactor). |
| `--completion-policy <P>` | `enforce` / `warn` / `off` — completion-readiness gating. |
| `--verification-policy <P>` | `auto` / `ask` / `off` — mutation verification. |
| `--agent-reliability-profile <P>` | `standard` / `strict`. |
| `--agent-speed-profile <P>` | `balanced` / `fast-safe`. |
| `--self-review` | Run a critic self-review before finishing. |

### Permissions & safety

| Flag | Description |
| --- | --- |
| `--permission-mode <M>` | `default` / `accept-edits` / `bypass-permissions` / `plan`. |
| `--dangerously-skip-permissions` | Bypass all permission checks. |
| `--dry-run` | Preview writes without changing files. |
| `--allowed-tools <LIST>` | Tool allowlist. |
| `--disallowed-tools <LIST>` | Tool denylist. |

### System prompt & context

| Flag | Description |
| --- | --- |
| `-s`, `--system-prompt <TEXT>` | Replace the system prompt. |
| `--system-prompt-file <PATH>` | Load the system prompt from a file. |
| `--append-system-prompt <TEXT>` | Append to the default system prompt. |
| `--no-claude-md` | Don't load `AGENTS.md` context. |
| `--dump-system-prompt` | Print the assembled system prompt and exit. |
| `--cwd <PATH>` | Working directory. |
| `--add-dir <PATH>` | Grant access to an additional directory (repeatable). |

### Config & integrations

| Flag | Description |
| --- | --- |
| `--settings <JSON|PATH>` | Override settings inline or from a file. |
| `--mcp-config <JSON>` | Inline MCP server definitions. |
| `--bare` | No hooks, plugins, or `AGENTS.md`. |
| `--disable-slash-commands` | Disable slash commands. |
| `--no-auto-compact` | Disable automatic context compaction. |
| `--vault-passphrase <PASS>` / `--vault-prompt` | Unlock the secret vault. |

### Output & logging

| Flag | Description |
| --- | --- |
| `--output-format <FMT>` | `text` / `json` / `stream-json`. |
| `--input-format <FMT>` | `text` / `stream-json`. |
| `-v`, `--verbose` | Enable verbose tracing. |

> This table summarizes the most common flags. The struct in `main.rs` is the
> authoritative and complete list — run `mangocode --help` for the exact set
> compiled into your binary.

---

## Environment variables

A few of the most relevant variables (full list in
[configuration.md](configuration.md#environment-variables)):

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, … | Provider credentials. |
| `RUST_LOG` | Tracing filter (e.g. `mangocode=debug`). |
| `MANGOCODE_FLAG_<NAME>` | Override a runtime feature flag. |
| `MANGOCODE_FEATURE_<NAME>` | Environment feature gate. |
| `MANGOCODE_SIMPLE` / `--bare` | Bare mode (no hooks/plugins/memory). |
| `OLLAMA_HOST`, `LM_STUDIO_HOST` | Local model server endpoints. |
