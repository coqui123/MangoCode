# MangoCode Spec Index (Rust Source)

All docs in this folder are based on ../src-rust.

## Document map

- 00_overview.md: workspace-level architecture summary
- 01_core_entry_query.md: CLI entry and query loop
- 02_commands.md: slash command framework
- 03_tools.md: tool architecture and tool families
- 04_components_core_messages.md: TUI core and message rendering
- 05_components_agents_permissions_design.md: agents/permissions/UI modules
- 06_services_context_state.md: core/api/mcp/bridge/query state services
- 07_hooks.md: Rust reactive equivalents (non-React)
- 08_ink_terminal.md: ratatui terminal rendering architecture
- 09_bridge_cli_remote.md: remote bridge and CLI integration
- 10_utils.md: utility modules in core
- 11_special_systems.md: memory/skills/plugins/voice/buddy
- 12_constants_types.md: canonical constants and schema types
- 13_rust_codebase.md: crate map and source-of-truth reference

## Source-of-truth policy

If a statement in spec conflicts with Rust source, Rust source wins.