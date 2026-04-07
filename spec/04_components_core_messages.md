# TUI Core + Message Rendering (Rust)

Source files:
- src-rust/crates/tui/src/lib.rs
- src-rust/crates/tui/src/messages/*
- src-rust/crates/tui/src/render.rs

## UI stack

MangoCode interactive UI is ratatui + crossterm based.

It includes:
- alternate screen rendering
- raw mode input
- mouse + bracketed paste support
- structured overlays/dialogs

## Message rendering

Message renderers live under tui/src/messages.

UI handles:
- user/assistant text blocks
- tool use + result rendering
- markdown/markdown_enhanced paths
- notifications and status banners

## Input + prompt

Prompt system includes:
- slash command parsing
- vim mode support
- history/typeahead/paste handling
- clipboard integration helpers

## Dialog/overlay capabilities

Core overlays include:
- diff viewer
- session browser/branching
- model picker
- stats dialog
- settings/privacy screens
- MCP view
- onboarding and auth dialogs