# MangoCode Documentation

MangoCode is a terminal-native AI coding assistant written in Rust. It pairs an
interactive terminal UI with an agentic query loop that streams model responses,
executes tools (file edits, shell, web, MCP, sub-agents), manages long-running
context, and persists session history — across more than 40 model providers.

This directory is the source-of-truth documentation for the MangoCode codebase.
Everything here describes the Rust workspace under [`src-rust/`](../src-rust).

> **Source-of-truth policy.** Where this documentation and the source disagree,
> the source wins. These pages are written against the crates as they exist in
> the workspace and use real type, function, module, and file names so you can
> jump straight from a doc to the code.

---

## What MangoCode is

- **Interactive TUI** built on [`ratatui`](https://ratatui.rs) + `crossterm`:
  streaming transcript, slash commands, vim-style prompt editor, inline images
  (Kitty / iTerm / Sixel), and ~40 overlays (model picker, diff viewer, session
  browser, MCP browser, stats, and more).
- **Agentic query loop** (`mangocode-query`): request → stream → dispatch tools →
  feed results back, with automatic context compaction, cost/turn budgets,
  completion/verification gates, sub-agents, and background schedulers.
- **Tool framework** (`mangocode-tools`): ~65 built-in tools spanning filesystem,
  shell, search, web/research, browser & computer-use, planning/tasks,
  coordination, MCP, skills, and code intelligence — each with a permission model.
- **Multi-provider API layer** (`mangocode-api`): one normalized interface adapting
  Anthropic, OpenAI (+ Codex), Google/Vertex, Azure, Bedrock, Cohere, GitHub
  Copilot, MiniMax, and 30+ OpenAI-compatible providers (Ollama, LM Studio,
  Groq, DeepSeek, OpenRouter, …).
- **Slash command system** (`mangocode-commands`): 100+ built-in commands plus
  user/plugin-defined templates.
- **Extensibility**: MCP client (`mangocode-mcp`), plugin runtime
  (`mangocode-plugins`), and an Agent Client Protocol server (`mangocode-acp`) for
  editor/IDE integration.
- **Remote control** (`mangocode-bridge`): drive a local session from the web UI.

---

## Documentation map

### Guides

| Page | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | Workspace layout, crate dependency graph, the turn loop, core data types, and on-disk layout. |
| [getting-started.md](getting-started.md) | Prerequisites, building, run modes, the full CLI argument reference, and authentication. |
| [configuration.md](configuration.md) | `settings.json`, environment variables, hooks, agents, providers, and feature flags. |
| [providers-and-models.md](providers-and-models.md) | Every supported provider, how models are resolved, and the model registry. |
| [slash-commands.md](slash-commands.md) | The complete catalog of built-in slash commands. |
| [tools.md](tools.md) | The tool framework, the permission model, and every built-in tool. |
| [memory-skills-plugins.md](memory-skills-plugins.md) | Memory (`AGENTS.md` + memory dirs + layered store), skills, plugins, MCP, and coordination. |

### Crate reference

Per-crate API references live in [`crates/`](crates/):

| Crate | Page | Role |
| --- | --- | --- |
| `mangocode` (cli) | [crates/cli.md](crates/cli.md) | Binary entrypoint, argument parsing, startup wiring, OAuth flows. |
| `mangocode-query` | [crates/query.md](crates/query.md) | The agentic turn loop and its subsystems. |
| `mangocode-tools` | [crates/tools.md](crates/tools.md) | Tool trait, context, permission model, and tool implementations. |
| `mangocode-api` | [crates/api.md](crates/api.md) | Provider abstraction, streaming, transformers, model registry. |
| `mangocode-core` | [crates/core.md](crates/core.md) | Shared types, config, storage, auth, memory, git, safety classifiers. |
| `mangocode-tui` | [crates/tui.md](crates/tui.md) | Terminal UI: app state, rendering, input, overlays. |
| `mangocode-commands` | [crates/commands.md](crates/commands.md) | Slash command framework and built-ins. |
| `mangocode-mcp` | [crates/mcp.md](crates/mcp.md) | Model Context Protocol client. |
| `mangocode-plugins` | [crates/plugins.md](crates/plugins.md) | Plugin manifest, loader, hooks, registry. |
| `mangocode-bridge` | [crates/bridge.md](crates/bridge.md) | Remote-control bridge. |
| `mangocode-acp` | [crates/acp.md](crates/acp.md) | Agent Client Protocol server. |
| support crates | [crates/support-crates.md](crates/support-crates.md) | `tool-runtime`, `file-search`, `turn-diff`, `sleep-inhibitor`. |

### Feature deep-dives

| Page | What it covers |
| --- | --- |
| [mango-intelligence.md](mango-intelligence.md) | Smart Attachments, Mango Research, output compression, and layered memory. |
| [chrome-remote-debug.md](chrome-remote-debug.md) | Driving Chrome/Chromium via `/chrome` and the DevTools Protocol. |

---

## Quick start

```bash
# Build the release binary (from the workspace root)
cd src-rust
cargo build --release

# Run it
./target/release/mangocode            # interactive TUI
./target/release/mangocode -p "..."   # headless / print mode
```

See [getting-started.md](getting-started.md) for prerequisites, run modes, the
full CLI surface, and how to authenticate a provider.

---

## Project facts

- **Language / edition:** Rust 2021.
- **Workspace:** Cargo, resolver v2, 15 member crates (see
  [`src-rust/Cargo.toml`](../src-rust/Cargo.toml)).
- **Binary name:** `mangocode` (crate `cli`).
- **Async runtime:** `tokio` (multi-threaded).
- **License:** Proprietary, source-available — free for personal use; commercial
  use requires a paid license. See [`LICENSE`](../LICENSE). Not an SPDX/OSI
  license.
- **Config home:** `~/.mangocode/` (settings, auth, vault, history, flags, usage).
- **Transcript home:** `~/.claude/projects/<encoded-project>/<session>.jsonl`.
