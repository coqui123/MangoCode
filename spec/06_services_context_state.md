# Services, Context, and State (Rust)

Primary source crates:
- src-rust/crates/core
- src-rust/crates/api
- src-rust/crates/mcp
- src-rust/crates/bridge
- src-rust/crates/query

## Core state and config

crates/core provides:
- Config/Settings/Theme/PermissionMode
- constants and shared types
- auth storage + oauth config helpers
- session/history storage (JSONL and SQLite)
- feature flags and analytics helpers

## API layer

crates/api provides:
- provider abstraction traits
- provider implementations (anthropic/openai/google/etc.)
- stream parsing + transformers
- model registry and effective model resolution
- retry/context-overflow error handling

## MCP layer

crates/mcp provides:
- MCP client protocol types
- stdio/http transports
- connection manager + status
- registry and oauth helpers

## Bridge layer

crates/bridge provides:
- bridge config
- jwt decode/expiry helpers
- remote session lifecycle and message protocol

## Query state machine

crates/query provides runtime state transitions for:
- conversation loop
- compaction
- away summary and memory extraction
- cron scheduler and command queue integration