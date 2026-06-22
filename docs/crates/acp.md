# Crate: `acp`

Path: [`src-rust/crates/acp`](../../src-rust/crates/acp) ·
Core: [`src/lib.rs`](../../src-rust/crates/acp/src/lib.rs)

`acp` implements an **Agent Client Protocol** server: JSON-RPC 2.0 over stdio that
lets an editor/IDE host drive MangoCode. The server runs MangoCode's native query
loop directly (local, no proxying) and streams session updates back to the host.

---

## Entry point

`run_acp_server()` reads JSON-RPC requests from stdin, dispatches them, and writes
responses/notifications to stdout. Requests and responses use
`JsonRpcRequest` / `JsonRpcResponse` / `JsonRpcError`.

---

## Methods

| Method | Purpose |
| --- | --- |
| `initialize` | Negotiate protocol version, capabilities, server info. |
| `authenticate` | Confirm authentication. |
| `tool/list` | List tools (optionally per session). |
| `model/list` | List available models. |
| `session/list` | List saved sessions. |
| `session/new` | Create a session (optional id, cwd, model, mode). |
| `session/load` | Load a saved session and stream its existing messages. |
| `session/prompt` | Send a user prompt; run the query loop and stream updates. |
| `session/cancel` | Cancel a running session. |
| `session/close` | Close a session. |
| `session/set_model` / `session/set_mode` | Change model / mode. |

---

## State & runtime

- `AcpServerState` holds the active `sessions`, the `running` cancellation tokens,
  and `pending_permissions`.
- `AcpSession` wraps a `ConversationSession` plus `cwd`, `model_override`, and
  `permission_mode`.
- `QueryRuntime` bundles the Anthropic client, tools, tool context, query config,
  and cost tracker used to run prompts.

### Permissions

`AcpPermissionHandler` implements the tool `PermissionHandler` trait. Based on the
session's `PermissionMode` it allows/denies automatically (e.g. `AcceptEdits`
allows writes, `BypassPermissions` allows all, `Plan` is read-only) or asks the
host via a `session/request_permission` notification and awaits the response.

### Event mapping

`acp_update_from_harness_event(...)` maps internal harness events to ACP session
updates: `message.delta` → message chunk; `tool.started`/`tool.completed` → tool
call; `file.changed` → diff; `permission.*` → permission; `turn.*` → status.

---

## Features

`acp` mirrors the per-tool feature flags fanned out from `cli` so its tool list
matches the rest of the build.
