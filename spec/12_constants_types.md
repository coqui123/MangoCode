# Constants, Types, and Schemas (Rust)

Primary source:
- src-rust/crates/core/src/lib.rs
- src-rust/crates/core/src/constants (via module exports)
- src-rust/crates/core/src/config/types modules
- src-rust/crates/api/src/types
- src-rust/crates/mcp/src/types

## Canonical type roots

Core exports include:
- ContentBlock, Message, Role
- ToolDefinition, ToolResultContent
- UsageInfo and cost/session types
- Config, Settings, Theme, PermissionMode
- ProviderId and ModelId newtypes

## Error model

Core defines a shared typed error enum used across crates.
Treat core error definitions as authoritative.

## API wire schemas

api crate defines request/response payload structs for:
- streaming model messages
- tool schema serialization
- provider-normalized message formats

## MCP schemas

mcp crate defines JSON-RPC and MCP protocol structs for:
- initialize handshake
- tools/resources/prompts metadata
- call/request/response payloads

Use current Rust struct/enum definitions as authoritative schema docs.