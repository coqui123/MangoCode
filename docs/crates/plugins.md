# Crate: `plugins`

Path: [`src-rust/crates/plugins`](../../src-rust/crates/plugins) ·
Modules: `lib.rs`, `manifest.rs`, `loader.rs`, `plugin.rs`, `registry.rs`,
`marketplace.rs`, `hooks.rs`

`plugins` is the plugin runtime: it discovers and loads plugins, parses manifests,
registers their contributions, enforces capabilities, and dispatches lifecycle
hooks. See [memory-skills-plugins.md](../memory-skills-plugins.md#plugins) for the
user view.

---

## Plugin layout

A plugin is a directory with a manifest and content subdirectories:

```
my-plugin/
├── plugin.json          # or plugin.toml
├── commands/            # *.md slash commands
├── agents/              # *.md agent definitions
├── skills/              # SKILL.md folders
├── hooks/hooks.json     # lifecycle hooks
├── output-styles/       # *.md / *.json styles
└── .mcp.json            # MCP servers (optional)
```

Plugins are discovered from `~/.mangocode/plugins/`, project `.mangocode/plugins/`,
and extra paths.

---

## Manifest (`manifest.rs`)

`PluginManifest` fields (parsed from JSON or TOML; camelCase normalized):

- **Identity** — `name` (required), `version`, `description`, `author`
  (`PluginAuthor`), `homepage`, `repository`, `license`, `keywords`,
  `marketplace_id`.
- **Contributions** — `commands`, `agents`, `skills`, `output_styles` (relative
  paths), `mcp_servers` (`PluginMcpServer`), `lsp_servers` (`PluginLspServer`),
  `hooks`.
- **User config** — `user_config` map of `PluginUserConfigOption`
  (typed: String/Number/Boolean/Directory/File, with title/description/required/
  default/sensitive).
- **Capabilities** — `capabilities: Option<Vec<String>>`:
  - `None` → legacy/trusted (allowed everything).
  - `Some([])` → explicitly nothing.
  - `Some([...])` → enforced list from `read_files`, `write_files`, `network`,
    `shell`, `browser`, `mcp`.

`from_json` / `from_toml` parse it; `validate()` checks the name and guards against
path traversal.

---

## Loading & registry (`lib.rs`, `loader.rs`, `registry.rs`)

- `load_plugins(project_dir, extra_paths) -> PluginRegistry` discovers and loads
  plugins (errors collected, registry still usable).
- `reload_plugins(...)` reloads and returns a `ReloadDiff`.
- Global registries (`GLOBAL_PLUGIN_REGISTRY`, `GLOBAL_HOOK_REGISTRY`) hold the
  active set; `skills_config_with_plugin_paths(...)` merges enabled plugin skill
  paths into skill discovery.
- `install_plugin_from_path(...)` validates a manifest and copies a plugin into
  `~/.mangocode/plugins/<name>`.
- `PluginSubCommand` + `parse_plugin_args(...)` back the `/plugin` command
  (List / Enable / Disable / Info / Install / Reload), with formatting helpers.

Capability enforcement happens at command registration via
`check_plugin_capability(...)`.

---

## Hooks (`hooks.rs`)

Plugins register shell hooks keyed by `HookEventKind`. `HookEventKind` covers a
broad event set: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`PermissionDenied`, `Notification`, `UserPromptSubmit`, `SessionStart`,
`SessionEnd`, `Stop`/`StopFailure`, `SubagentStart`/`SubagentStop`,
`PreCompact`/`PostCompact`, `PermissionRequest`, `Setup`, `TeammateIdle`,
`TaskCreated`/`TaskCompleted`, `Elicitation`/`ElicitationResult`, `ConfigChange`,
`WorktreeCreate`/`WorktreeRemove`, `InstructionsLoaded`, `CwdChanged`,
`FileChanged`.

- `RegisteredHook` = command + optional `matcher` (glob) + `blocking` + plugin
  identity. `register_plugin_hooks(...)` adds them; `prune_hooks(...)` removes hooks
  for disabled/uninstalled plugins.
- `run_hook_sync(hook, event_json)` spawns the command with `CLAUDE_PLUGIN_ROOT` /
  `CLAUDE_PLUGIN_NAME` set and the event JSON on stdin, returning a `HookOutcome`
  (`Allow` / `Deny(reason)` / `Error(msg)`). `glob_match` supports `*` wildcards.
- The query loop calls `run_global_pre_tool_hook` / `run_global_post_tool_hook` /
  `run_global_user_prompt_submit_hook` / `run_global_lifecycle_hook`; pre-tool and
  prompt-submit hooks can block, post-tool hooks are informational.

`marketplace.rs` provides marketplace metadata/integration for discovering plugins.
