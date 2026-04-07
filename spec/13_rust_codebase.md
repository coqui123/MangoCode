# Rust Codebase Reference (Canonical)

Source root: ../src-rust

## Workspace crates

- core
- api
- tools
- query
- tui
- commands
- mcp
- bridge
- cli
- buddy
- plugins
- acp

## Binary

CLI binary entrypoint:
- crates/cli/src/main.rs

## Key architecture facts

- Cargo workspace resolver v2
- Rust 2021 edition
- Async runtime: tokio
- Terminal UI: ratatui + crossterm
- Tool protocol implemented in Rust trait system
- Provider abstraction supports multiple backends

## Where to inspect features

- Command behavior: crates/commands
- Tool implementations: crates/tools
- Query/turn logic: crates/query
- Shared types/config: crates/core
- Streaming providers: crates/api
- MCP client: crates/mcp
- Remote bridge: crates/bridge
- UI behavior: crates/tui

This file is the high-level map used by all other spec files.