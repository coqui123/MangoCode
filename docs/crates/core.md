# Crate: `core`

Path: [`src-rust/crates/core`](../../src-rust/crates/core) ·
Core: [`src/lib.rs`](../../src-rust/crates/core/src/lib.rs)

`core` is the foundation every other crate depends on. It defines the shared data
model, configuration, error type, and a large set of subsystems: storage, auth,
memory, git, safety classifiers, prompt assembly helpers, and utilities. This page
groups the modules by area.

---

## Shared types & errors (`lib.rs`)

- **`Message`** / **`MessageContent`** / **`Role`** — the conversation unit.
- **`ContentBlock`** — the content union (`Text`, `Image`, `ToolUse`, `ToolResult`,
  `Thinking`, `RedactedThinking`, `Document`, and MangoCode blocks like
  `UserLocalCommandOutput`, `UserCommand`, `UserMemoryInput`, `SystemAPIError`,
  `CollapsedReadSearch`, `TaskAssignment`).
- **`ToolDefinition`**, **`ToolResultContent`**.
- **`UsageInfo`** / **`MessageCost`** — token and cost accounting.
- **`ClaudeError`** — the shared error enum with `Result<T>` alias,
  `is_retryable()`, and `is_context_limit()`.
- **`Config`** and its nested types — the merged runtime configuration (see
  [configuration.md](../configuration.md)): `AgentDefinition`, `ProviderConfig`,
  `HookEvent`/`HookEntry`, `PermissionMode`, `AgentCompletionPolicy`,
  `VerificationPolicy`, `ResearchConfig`, `AttachmentConfig`, `MemoryConfig`, etc.

### Identifiers & budgets

- **`provider_id.rs`** — `ProviderId` / `ModelId` newtypes with constants for every
  known provider and safe `provider/model` parsing.
- **`effort.rs`** — `EffortLevel` (`Low`/`Medium`/`High`/`Max`) with thinking
  budgets, temperature, and UI glyphs.
- **`token_budget.rs`** — `TokenBudget` / `TokenWarningLevel`, `should_compact()`,
  formatting, and `context_window_for_model()`.

---

## Configuration & settings

- **`migrations.rs`** — idempotent `settings.json` upgrades applied on startup
  (model alias renames, permission-mode moves, etc.); `run_migrations(&mut Value)`.
- **`settings_sync.rs`** — sync `settings.json` and `AGENTS.md` with claude.ai over
  OAuth (user- and project-scope, ETag-based).
- **`remote_settings.rs`** — enterprise/remote managed settings, cached and polled.
- **`feature_flags.rs`** — persisted runtime flags (`~/.mangocode/flags.json`) with
  `MANGOCODE_FLAG_*` overrides; constants like `FLAG_PLAN_SEARCH`,
  `FLAG_EXECUTION_SCRATCHPAD`, `FLAG_SELF_REVIEW`.
- **`feature_gates.rs`** — env-var feature gates (`MANGOCODE_FEATURE_*`), dynamic
  config, and bare mode.

---

## Auth & security

- **`auth_store.rs`** — `AuthStore` over `~/.mangocode/auth.json` (`StoredCredential`
  = API key or OAuth token), with env-var fallback and optional vault mirroring.
- **`vault.rs`** — AES-256-GCM encrypted vault (`~/.mangocode/vault.enc`) with an
  Argon2id-derived key; session passphrase cache (zeroized). Also gateway and
  Pipedream config.
- **`oauth_config.rs`** — Claude OAuth endpoints, scopes, PKCE helpers, auth-URL
  builder, and profile fetch.
- **`codex_oauth.rs`** — OpenAI Codex OAuth constants and helpers (model
  normalization, JWT decode, account-id extraction).
- **`device_code.rs`** — RFC 8628 device-code flow (e.g. GitHub).
- **`crypto_utils.rs`** — SHA-256, base64url, UUID, work-secret, project-root
  encoding.

---

## Session & transcript storage

- **`session_storage.rs`** — append-only JSONL transcripts at
  `~/.claude/projects/<encoded-root>/<session>.jsonl`. `TranscriptEntry` variants
  (User/Assistant/Attachment/System/Summary/AiTitle/CustomTitle/LastPrompt/
  Tombstone), `write_transcript_entry`, `load_transcript`, `list_sessions`,
  soft-delete via tombstones, and fast tail reads.
- **`sqlite_storage.rs`** — SQLite-backed session store (`SqliteSessionStore`).
- **`prompt_history.rs`** — `~/.mangocode/history.jsonl` with externalized large
  pastes under `pastes/` and advisory locking.
- **`remote_session.rs` / `cloud_session.rs` / `session_share.rs`** — cloud sync
  (REST + WebSocket), message adapters, and session sharing/export.
- **`usage_ledger.rs`** — cumulative usage (`~/.mangocode/usage.json`); per-session
  records and rolling cost queries.
- **`session_tracing.rs`** — OpenTelemetry span stubs (no-ops unless the `otel`
  feature is built).
- **`harness.rs`** — durable turn events and git/file **checkpoints**
  (`~/.mangocode/sessions.db` + `traces/`), enabling `/undo` and `/rewind`.

---

## Prompt & system assembly

- **`system_prompt.rs`** — assembling and wrapping the system prompt, including
  `wrap_untrusted_content(...)` for untrusted tool input.
- **`output_styles.rs`** — built-in and user/plugin output styles injected into the
  prompt.
- **`harness.rs`**, **`tips.rs`**, **`status_notices.rs`** — session events, tips,
  and status notices.

---

## Memory

- **`claudemd.rs`** — `AGENTS.md` hierarchy (Managed/User/Project/Local),
  frontmatter parsing, `@include` expansion, and prompt assembly.
- **`memdir.rs`** — file-based memory directory scanning, `MEMORY.md` index,
  freshness notes, and simple relevance matching.
- **`layered_memory.rs`** — SQLite semantic memory store (lexical + sparse + dense
  embeddings) with privacy filtering.
- **`team_memory_sync.rs`** — ETag-based delta sync of team memory.
- **`frontmatter.rs`** — shared zero-copy YAML frontmatter splitting.

See [memory-skills-plugins.md](../memory-skills-plugins.md) for the full picture.

---

## Skills

- **`skill_discovery.rs`** — discovers markdown skills (with triggers,
  dependencies, sub-files, scripts, QA) from project/user/extra/git paths.

---

## Git & files

- **`git_utils.rs`** — repo root, current branch, modified files, staged/unstaged
  diffs, commit history.
- **`git_diff.rs`** — robust diff path parsing (quoted paths, octal escapes,
  renames).
- **`file_history.rs`** — per-session file modification tracking (for `/rewind`).
- **`snapshot.rs`** — per-tool-call content snapshots (for `/undo`).
- **`attachments.rs` / `smart_attachments.rs`** — context attachments and
  document→markdown routing based on model capabilities.

---

## Text & formatting

- **`truncate.rs`** — UTF-8-safe truncation helpers.
- **`message_utils.rs`** — token estimation, message text extraction, merging,
  external-value conversion (strips transcript-only metadata for uploads).
- **`format_utils.rs`** — cost / duration / token / relative-time formatting.
- **`context_collapse.rs`** — conversation-size reduction strategies and token
  estimation.

---

## Safety classifiers & permissions

- **`bash_classifier.rs`** — `BashRiskLevel` (`Safe`→`Critical`); detects fork
  bombs, pipe-to-shell, `rm -rf /`, `dd`/`mkfs`/`shred`, etc.
- **`ps_classifier.rs`** — `PsRiskLevel`; detects IEX-download RCE, WebClient
  exec, destructive `Remove-Item`, disk wipes, critical service stops.
- **`permission_critic.rs`** — optional LLM permission evaluator (cheap model,
  cached) with classifier fallback.

---

## Coordination, goals & mode

- **`coordination.rs`** — local SQLite session presence, work claims, and messages
  for multi-process coordination; conflict warnings.
- **`goals.rs`** — persistent thread goals with token budgets and status; system
  prompt injection.
- **`auto_mode.rs`** — auto-approve mode state (`AcceptEdits` / `BypassPermissions`
  / `Plan`) and opt-in tracking.

---

## Integrations & misc

- **`ide.rs`** — IDE detection (VS Code, Cursor, Windsurf, JetBrains, Zed, …).
- **`lsp.rs`** — language-server integration backing the `LSP` tool.
- **`voice.rs`** — voice support types/events (the TUI captures and transcribes).
- **`mcp_templates.rs`** — MCP tool-definition templates.
- **`command_line.rs`** — shell command-line word splitting (quotes/escapes).
- **`chrome_js.rs`** — JS injection helpers for browser automation.
- **`keybindings.rs`** — keybinding parsing and contexts (`~/.claude/keybindings.json`).
- **`analytics.rs`** — atomic session metrics.
- **`update_check.rs`** — update checks.
