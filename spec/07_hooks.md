# Hooks and Reactive Behaviors (Rust Mapping)

Rust source does not use React hooks.

This spec maps equivalent reactive behavior to Rust modules.

## Equivalent systems

Input/event reactions:
- tui prompt input and crossterm event loop

Permission reactions:
- tools::ToolContext permission checks
- tui permission dialogs

Command reactions:
- commands::SlashCommand dispatch and CommandResult handling

Background reactions:
- query command queue drain
- query cron scheduler
- bridge polling/reconnect loops

State-derived rendering:
- tui app state and overlay render functions

Use Rust module behavior as source of truth; ignore prior React hook docs.