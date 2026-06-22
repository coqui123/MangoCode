# Crate Reference

Per-crate API references for the MangoCode workspace
([`src-rust/`](../../src-rust)). For the big picture and how the crates fit
together, start with [../architecture.md](../architecture.md).

| Crate | Page | Role |
| --- | --- | --- |
| `cli` (bin `mangocode`) | [cli.md](cli.md) | Binary entrypoint, argument parsing, startup wiring, OAuth flows. |
| `query` | [query.md](query.md) | The agentic turn loop and its subsystems. |
| `tools` | [tools.md](tools.md) | Tool trait, context, permission model, and tool implementations. |
| `api` | [api.md](api.md) | Provider abstraction, streaming, transformers, model registry. |
| `core` | [core.md](core.md) | Shared types, config, storage, auth, memory, git, safety classifiers. |
| `tui` | [tui.md](tui.md) | Terminal UI: app state, rendering, input, overlays. |
| `commands` | [commands.md](commands.md) | Slash command framework and built-ins. |
| `mcp` | [mcp.md](mcp.md) | Model Context Protocol client. |
| `plugins` | [plugins.md](plugins.md) | Plugin manifest, loader, hooks, registry. |
| `bridge` | [bridge.md](bridge.md) | Remote-control bridge. |
| `acp` | [acp.md](acp.md) | Agent Client Protocol server. |
| `tool-runtime`, `file-search`, `turn-diff`, `sleep-inhibitor` | [support-crates.md](support-crates.md) | Small foundational crates. |

> The `tools` crate has a low-level API reference ([tools.md](tools.md)) and a
> user-facing guide with the full tool catalog ([../tools.md](../tools.md)).
