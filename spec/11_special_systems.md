# Special Systems (Rust)

Source crates:
- src-rust/crates/core
- src-rust/crates/query
- src-rust/crates/plugins
- src-rust/crates/buddy
- src-rust/crates/tui

## Memory system

Implemented via core + query:
- memory file loading module in core
- session memory extraction/state
- away summary and auto dream helpers

## Plugins and skills

Plugins crate handles:
- manifest parsing
- plugin loader
- registry/marketplace wiring

Skills pipeline spans:
- core skill discovery
- tools skill tool
- query skill prefetch

## Voice

Voice-related support appears in:
- core voice module
- tui voice capture and notice modules

## Buddy

buddy crate exists as dedicated subsystem.

## Keybindings/themes

Managed in core keybindings + tui theme modules.