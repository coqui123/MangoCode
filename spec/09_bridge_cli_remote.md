# Bridge, CLI Remote, and Session Transport (Rust)

Source files:
- src-rust/crates/bridge/src/lib.rs
- src-rust/crates/cli/src/main.rs

## Purpose

Bridge connects local MangoCode session to remote web control.

## Core responsibilities

- bridge configuration from env/defaults
- token and JWT claim parsing
- device fingerprint generation
- bridge poll loop and reconnect handling
- permission decision and session event routing

## Integration points

- CLI starts/stops bridge in supported modes
- TUI reflects bridge state through bridge_state module
- session lifecycle and timeout values enforced by bridge config

## Key environment integration

Bridge config supports env overrides for:
- server URL
- session/access token
- bridge enablement

Exact variable handling lives in bridge crate source.