# Crate: `bridge`

Path: [`src-rust/crates/bridge`](../../src-rust/crates/bridge) ·
Core: [`src/lib.rs`](../../src-rust/crates/bridge/src/lib.rs)

`bridge` connects a local MangoCode session to the web UI for **remote control**.
It registers a session with the server, long-polls for inbound messages (prompts,
permission responses, cancellations), and uploads outbound events (text deltas,
tool start/end, turn completion). It is interactive-only (not used in headless
mode).

---

## Configuration

`BridgeConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Master toggle. |
| `server_url` | `https://claude.ai` | Server base URL. |
| `device_id` | device fingerprint | Stable per-machine id. |
| `session_token` | `None` | Bearer / session-ingress token. |
| `polling_interval_ms` | `1000` | Poll cadence. |
| `max_reconnect_attempts` | `10` | Failures before giving up. |
| `session_timeout_ms` | 24h | Inactivity timeout. |
| `runner_version` | crate version | Reported client version. |

`from_env()` reads `MANGOCODE_BRIDGE_URL` / `CLAUDE_BRIDGE_BASE_URL` (sets
`enabled`) and `MANGOCODE_BRIDGE_TOKEN` / `CLAUDE_BRIDGE_OAUTH_TOKEN`.
`is_active()` requires `enabled` + a token. `validate_id(...)` rejects ids that
could enable path traversal.

`device_fingerprint()` is a SHA-256 of hostname + username + home dir (stable per
machine). JWT helpers (`JwtClaims::decode`, `is_expired`, `decode_jwt_expiry`,
`jwt_is_expired`) parse session-ingress tokens client-side (no signature check;
strips the `sk-ant-si-` prefix).

---

## Protocol

**Inbound** (`BridgeMessage`): `UserMessage { content, attachments, … }`,
`PermissionResponse { request_id, tool_use_id, decision }`, `Cancel`, `Ping`.
`PermissionDecision` = `Allow` / `AllowPermanently` / `Deny` / `DenyPermanently`.

**Outbound** (`BridgeEvent`): `TextDelta`, `ToolStart`, `ToolEnd`,
`PermissionRequest`, `TurnComplete { usage }`, `Error`, `Pong`, `SessionState`.

REST endpoints used: register/deregister a session, `poll` for messages, and
`events` to upload outbound events. A higher-level API
(`start_bridge_session` / `poll_bridge_messages` / `post_bridge_response` /
`post_bridge_event`) covers the simpler message-relay flow.

---

## Session lifecycle

`BridgeSession` owns config, a session UUID, connection state
(`Disconnected`/`Connecting`/`Connected`/`Running`/`Error`), and an HTTP client. It
provides `register` / `deregister`, `poll_messages`, `upload_events`, and the
`run_poll_loop` that drains outbound events, polls for inbound messages, forwards
them over channels, and reconnects with exponential backoff (capped) up to
`max_reconnect_attempts`, deregistering on shutdown.

---

## Integration

`run_bridge_loop(config, tui_tx, outbound_rx, cancel)` is the high-level entry the
TUI uses: it registers the session, emits a `Connected { session_url, session_id }`
event, spawns the poll loop, and translates between protocol messages and TUI
events:

- `TuiBridgeEvent` — `Connected`, `Disconnected`, `Reconnecting`, `InboundPrompt`,
  `Cancelled`, `PermissionResponse`, `SessionNameUpdate`, `Error`, `Ping`.
- `BridgeOutbound` — what the query loop sends out: `TextDelta`, `ToolStart`,
  `ToolEnd`, `TurnComplete`, `Error`, `SessionMeta`.

The TUI reflects connection state via `tui::bridge_state`. The `cli` crate decides
whether to start the bridge with `resolve_bridge_config(...)` (see
[cli.md](cli.md#bridge-integration)).
