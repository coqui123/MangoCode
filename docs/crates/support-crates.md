# Support crates

Four small, focused crates underpin the larger ones.

---

## `tool-runtime`

Path: [`src-rust/crates/tool-runtime`](../../src-rust/crates/tool-runtime) ·
Modules: `lib.rs`, `fuzzy.rs`

Shared types and logic for describing and scheduling tools, used by `tools` and
the query loop.

- **`ToolHandlerKind`** — `BuiltIn` / `Mcp` / `Plugin` / `Alias` / `Unavailable`.
- **`ToolCapabilities`** — `mutating`, `parallel_safe`, `affected_paths`,
  `network_targets`, `approval_keys`, `aliases`, `sandbox_preference`,
  `supports_cancellation`, `output_policy`. Built with `read_only()` / `mutating()`
  + builder methods.
- **`ToolSpec`** — name, description, schema, handler kind, aliases, capabilities.
- **`ToolRegistryPlan`** — the resolved tool set with alias map;
  `canonical_name(requested)`, `spec_for(requested)`, `suggestions_for(requested, n)`.
- **`plan_execution_batches(calls)`** — groups tool calls into execution batches:
  parallel-safe reads run together; mutations and path/host conflicts are
  serialized.
- **`fuzzy.rs`** — `levenshtein` and `damerau_levenshtein` edit-distance functions
  used to suggest close tool names.

---

## `file-search`

Path: [`src-rust/crates/file-search`](../../src-rust/crates/file-search) ·
Module: `lib.rs`

File and code indexing for the prompt's `@file`/`@symbol`/`@recent` typeahead and
the `CodeSearch` tool.

- **`FileSearchIndex`** — built with `build(root)` / `build_limited(root, max)`.
  Skips common junk dirs (`.git`, `node_modules`, `target`/`dist`/`build`, virtualenvs,
  caches) and honors `.gitignore` / `.sembleignore`.
- **`SearchKind`** — `Any` / `File` / `Folder` / `Symbol` / `Recent`.
- **Filename search** — scored hits (`SearchHit`): exact > full-match >
  filename-contains > display-contains > fuzzy.
- **Code search** — `add_lightweight_symbols(...)` extracts functions/classes/
  structs; `add_code_chunks(...)` chunks files (~1500 chars); `search_code(query, n)`
  and `search_code_filtered(...)` rank with TF-IDF; `find_related(path, line, n)`
  uses token Jaccard similarity. Detects a broad set of languages and special files
  (Dockerfile, Makefile, CMakeLists, Gemfile, …).

---

## `turn-diff`

Path: [`src-rust/crates/turn-diff`](../../src-rust/crates/turn-diff) ·
Module: `lib.rs`

Per-turn change export and rollback, built on `core::file_history`.

- **`TurnPatch`** — one file's change for a turn (before/after text + existence,
  tool names, unified diff).
- **`PatchBundle`** — all patches for a turn, with a combined unified diff and file
  list.
- **`patches_for_turn(...)` / `export_patch_bundle(...)`** — build patches/bundles.
- **`unified_diff(...)` / `unified_diff_with_existence(...)`** — generate diffs
  (`/dev/null` for created/deleted files) via `similar`.
- **`rollback_turn(history, turn_index) -> RollbackResult`** — revert a turn's file
  changes (reports partial failures for binary files / missing snapshots).
- **`patch_id(turn_index, relative_path)`** — stable `turn-<n>-<digest>` id.

Backs `/rewind` and turn-scoped diffs in the diff viewer.

---

## `sleep-inhibitor`

Path: [`src-rust/crates/sleep-inhibitor`](../../src-rust/crates/sleep-inhibitor) ·
Modules: `lib.rs`, `windows_inhibitor.rs`, `macos.rs`, `iokit_bindings.rs`,
`linux_inhibitor.rs`, `dummy.rs`

Cross-platform prevention of OS idle sleep while a turn is running (gated by the
`prevent_idle_sleep` setting).

- **`SleepInhibitor`** — `new(enabled)`, `set_turn_running(bool)`,
  `set_enabled(bool)`, `is_turn_running()`. The platform inhibitor engages only when
  both `enabled` and a turn is running.
- **Windows** — Power Request API (`PowerCreateRequest` +
  `PowerSetRequest(PowerRequestSystemRequired)`).
- **macOS** — IOKit power assertion (`IOPMAssertionCreateWithName`,
  `PreventUserIdleSystemSleep`).
- **Linux** — spawns `systemd-inhibit` or `gnome-session-inhibit`, with the child
  set to die if the parent exits.
- **Other** — a no-op `dummy` backend.
