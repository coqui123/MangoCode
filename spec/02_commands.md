# Commands System (Rust)

Source files:
- src-rust/crates/commands/src/lib.rs
- src-rust/crates/commands/src/named_commands.rs

## Architecture

Slash commands are implemented as structs that implement SlashCommand.

Core types:
- CommandContext
- CommandResult
- SlashCommand trait

## Command execution model

Each command:
- exposes name, aliases, description/help
- parses arg string
- returns CommandResult for UI/runtime handling

CommandResult supports:
- local message output
- user message injection
- config mutation
- conversation clear/set
- session resume/rename
- OAuth flow trigger
- overlays (rewind/hooks)
- exit/silent/error

## Built-ins in crate

The command crate defines many built-ins including categories:
- session/control: help, clear, exit, resume, status
- model/config: model, config, theme, effort, output style
- analysis: diff, summary, review, insights
- integration: mcp, plugins, remote-control, connect
- utility: copy, files, export, stats, keybindings
- auth/plan: login, logout, plan, tasks, permissions

Spec rule: command behavior should be validated against current command source, not legacy docs.