# Crate: `mcp`

Path: [`src-rust/crates/mcp`](../../src-rust/crates/mcp) ·
Modules: `lib.rs`, `connection_manager.rs`, `oauth.rs`, `registry.rs`

`mcp` is MangoCode's client for the **Model Context Protocol** (protocol version
`2024-11-05`). It connects to MCP servers over stdio or HTTP/SSE, performs the
handshake, discovers tools/resources/prompts, and keeps connections alive. See
[memory-skills-plugins.md](../memory-skills-plugins.md#mcp-model-context-protocol)
for the user view.

---

## Protocol types

JSON-RPC 2.0 plus MCP messages:

- `JsonRpcRequest` / `JsonRpcResponse` / `JsonRpcError` (with notification
  constructors).
- Handshake: `InitializeParams` / `InitializeResult` (`ClientCapabilities`,
  `ServerCapabilities`, `ClientInfo` / `ServerInfo`, optional server `instructions`).
- Tools: `McpTool` (name, description, `input_schema`), `ListToolsResult`,
  `CallToolParams`, `CallToolResult` (`McpContent` = text / image / resource).
- Resources: `McpResource`, `ListResourcesResult`, `ResourceContents`.
- Prompts: `McpPrompt`, `GetPromptResult`, `PromptMessage`.

`expand_env_vars` / `expand_server_config` expand `${VAR}` and `${VAR:-default}`
in server configurations.

---

## Transports

The `McpTransport` trait abstracts message send/recv/close and notification
streaming. Two implementations:

- **`StdioTransport`** — spawns the server as a subprocess and talks over
  stdin/stdout, with a background reader forwarding lines.
- **`HttpTransport`** — HTTP/SSE for hosted servers, including Pipedream MCP OAuth
  with layered token resolution (explicit config → vault → env → `pipedream.json`
  → default) and the required MCP session/protocol headers.

---

## Client

`McpClient` holds the transport plus discovered `server_info`, `capabilities`,
`tools`, `resources`, `prompts`, and `instructions`. Key methods:

- `connect_stdio(config)` / `connect_remote(config)` then `initialize(self)` —
  perform the handshake (send `initialize`, then `notifications/initialized`) and
  auto-discover tools/resources/prompts.
- `list_tools()` / `call_tool(name, args)`.
- `list_resources()` / `read_resource(uri)`.

A background message router dispatches JSON-RPC responses to pending callers by id
and forwards notifications.

---

## Connection manager

`McpConnectionManager` owns a set of servers and their lifecycle:

- `connect_all()` / `connect(name)` / `disconnect(name)` / `restart(name)`.
- `all_statuses()` returns `McpServerStatus` per server (`Connected { tool_count }`,
  `Connecting`, `Disconnected { last_error }`, `Failed { error, retry_at }`).
- `client(name)` returns the live `Arc<McpClient>`.
- An automatic reconnect loop with exponential backoff (capped) recovers dropped
  connections.

The TUI's `/mcp` view renders these statuses and the discovered tools.

---

## OAuth (`oauth.rs`)

PKCE-based OAuth for MCP servers: `pkce_verifier` / `pkce_challenge`, browser login
(`initiate_xaa_login`), `exchange_code`, and `refresh_mcp_token`. Tokens are stored
per server under `~/.mangocode/mcp-tokens/<server>.json` (`store_mcp_token` /
`get_mcp_token` / `remove_mcp_token`). Surfaced to the model via the `mcp__auth`
tool.

---

## Registry (`registry.rs`)

A compile-time list of well-known official MCP servers (`OfficialMcpServer`:
name, description, homepage, install command, categories) with `search_registry`
and `find_server`. `prefetch_official_mcp_urls()` optionally refreshes the list
from the Anthropic MCP registry (skipped under
`MANGOCODE_DISABLE_NONESSENTIAL_TRAFFIC`); `is_official_mcp_url` checks a URL
against the cached set (fail-closed).
