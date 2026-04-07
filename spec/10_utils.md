# Utility and Support Modules (Rust)

Primary source:
- src-rust/crates/core/src/*.rs

## Utility domains

Formatting/text:
- format_utils, truncate, context_collapse

Git/filesystem:
- git_utils, file_history, snapshot

Auth/security:
- auth_store, oauth_config, crypto_utils, device_code

Session/state:
- session_storage, sqlite_storage, prompt_history, remote_session

Feature controls:
- feature_flags, feature_gates, status_notices

Memory and prompt support:
- memory markdown loader module, memdir, output_styles, system_prompt

This file tracks Rust utility modules only.