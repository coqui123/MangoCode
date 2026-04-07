# MangoCode - Master Architecture Overview (Rust Source of Truth)

Source root: ../src-rust
Primary language: Rust
Workspace: Cargo multi-crate

## What MangoCode is

MangoCode is a Rust CLI coding assistant with:
- Interactive terminal UI (ratatui + crossterm)
- Agentic query loop with streaming model responses
- Tool execution framework (file, shell, web, MCP, tasks, agents)
- Slash command system
- MCP client integration
- Remote bridge support
- Session/history/memory systems

## Repository of record

All architecture in this spec is based on:
- src-rust/Cargo.toml
- src-rust/crates/*

TypeScript references were removed. Rust is authoritative.

## Workspace layout

- crates/cli: binary entrypoint and CLI flow
- crates/query: turn loop, compaction, schedulers
- crates/tools: tool framework and tool implementations
- crates/commands: slash command framework
- crates/tui: terminal UI and overlays
- crates/core: config, types, constants, storage, auth, utilities
- crates/api: provider adapters + streaming + model registry
- crates/mcp: MCP client and connection manager
- crates/bridge: remote control bridge
- crates/plugins: plugin loading and registry
- crates/buddy: companion subsystem
- crates/acp: auxiliary protocol crate

## Runtime flow

1. CLI parses args in crates/cli/src/main.rs
2. Config and context built from crates/core
3. Query loop runs in crates/query
4. Tools dispatched through crates/tools
5. UI rendered by crates/tui (interactive mode)
6. Provider calls handled by crates/api
7. Optional MCP/bridge/plugins layered in

## Scope rule for all spec files

Every section in this spec directory must describe current Rust source under src-rust only.