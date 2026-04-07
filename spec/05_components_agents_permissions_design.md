# TUI Agents, Permissions, and Design Modules (Rust)

Source files:
- src-rust/crates/tui/src/agents_view.rs
- src-rust/crates/tui/src/dialogs.rs
- src-rust/crates/tui/src/bypass_permissions_dialog.rs
- src-rust/crates/tui/src/hooks_config_menu.rs
- src-rust/crates/tui/src/plugin_views.rs

## Agents UI

Agents view provides:
- agent definition listing
- coordinator status rendering
- agent status structs

## Permission UX

Permission and safety UX includes:
- generic permission dialogs
- bypass-permissions startup confirmation
- hooks config browser

## Feature modules in TUI

Current Rust TUI modules cover:
- plugin recommendations
- context visualization
- tasks overlay
- overage/upgrade flows
- release and onboarding screens

No React/TS component model is used in Rust UI.