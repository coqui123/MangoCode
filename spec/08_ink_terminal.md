# Terminal Rendering Architecture (Rust)

Source files:
- src-rust/crates/tui/src/lib.rs
- src-rust/crates/tui/src/app.rs
- src-rust/crates/tui/src/render.rs

## Stack

- Terminal control: crossterm
- Widget/layout/rendering: ratatui
- Backend: CrosstermBackend

## Lifecycle

1. setup_terminal(): raw mode + alt screen
2. App event loop processes input and updates state
3. render pass draws message list, prompt, overlays
4. restore_terminal() on exit

## Rendering concerns

- message virtualization/list behavior
- markdown and diff rendering paths
- overlay/dialog composition
- status and notification banners

This replaces prior Ink/React-terminal architecture references.