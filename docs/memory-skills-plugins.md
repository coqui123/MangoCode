# Memory, Skills, Plugins, MCP & Coordination

This page covers the cross-cutting systems that give MangoCode persistent context
and extensibility. Implementation lives mostly in `core`, with plugin and MCP
runtimes in their own crates.

---

## Memory

MangoCode has three complementary memory layers, all assembled into the system
prompt for each turn.

### 1. `AGENTS.md` hierarchy (`core::claudemd`)

Markdown instruction files loaded in priority order:

1. **Managed** — `~/.mangocode/rules/*.md`
2. **User** — `~/.mangocode/AGENTS.md`
3. **Project** — `<project>/AGENTS.md`
4. **Local** — `<project>/.mangocode/AGENTS.md`

Files support YAML frontmatter and `@include <path>` directives (recursive, with
depth and circular-include protection, size-capped per file). They're concatenated
into a single memory prompt block. Disable with `--no-claude-md` or
`disable_claude_mds`.

### 2. File-based memory directory (`core::memdir`)

A per-project memory directory (default
`~/.mangocode/projects/<sanitized-root>/memory/`) holds one fact per `.md` file
with frontmatter (`name`, `description`, `type` of user/feedback/project/reference).
`MEMORY.md` is the entrypoint index (truncated if large). A lightweight TF-IDF
match surfaces the most relevant memory files for a query, and freshness notes flag
stale entries. Auto-memory can be disabled via `MANGOCODE_DISABLE_AUTO_MEMORY`,
`MANGOCODE_SIMPLE`, `MANGOCODE_REMOTE`, or settings.

### 3. Layered semantic store (`core::layered_memory`)

An optional SQLite store (`.mangocode/layered-memory.sqlite`) with lexical + sparse
+ dense (embedding) retrieval over memory classes (`ProjectFact`,
`UserPreference`, `Decision`, `ExternalDoc`). Embeddings come from `fastembed`
(default `BAAI/bge-base-en-v1.5`) or an external command. Privacy filtering strips
`<private>…</private>` sections and detects secrets before storing. Enable via
`Config.memory.layered_retrieval`. Explicit captures parse `remember:`,
`decision:`, `preference:`, and URLs.

### Runtime memory subsystems

The query loop adds:

- **Session memory extraction** (`query::session_memory`) — extracts durable facts
  from the conversation into topic files.
- **Memory loader** (`query::memory_loader`) — loads `MEMORY.md` and relevant
  topics each turn.
- **Auto-dream consolidation** (`query::auto_dream`) — periodic background memory
  consolidation gated on time + new sessions + a lock.
- **Team memory sync** (`core::team_memory_sync`) — ETag-based delta sync of a
  shared team memory directory with claude.ai.

---

## Skills

Skills are reusable prompt templates discovered from disk and git
(`core::skill_discovery`, surfaced by the `Skill` tool and prefetched by
`query::skill_prefetch`). Search order:

1. Project `.mangocode/skills/` and `.mangocode/commands/` (walking up from cwd).
2. Project `.agents/skills/` and `.agents/commands/`.
3. User `~/.mangocode/skills/` and `~/.mangocode/commands/`.
4. Extra paths from `Config.skills.paths`.
5. Git-URL repos from `Config.skills.urls` (cloned and cached).
6. Bundled skills compiled into the binary (`tools::bundled_skills`).

A skill is a markdown file (or a folder with `SKILL.md` plus sub-files and a
`scripts/` directory) with frontmatter:

| Field | Meaning |
| --- | --- |
| `name` | Skill name (defaults to file stem). |
| `description` | One-line description. |
| `triggers` | Keyword phrases that auto-load the skill. |
| `dependencies` | Other skills to load first. |
| `qa_required` / `qa_steps` | Mandatory QA gating and steps. |

Templates support `$ARGUMENTS`, `$1`, `$2` substitution. Run with `/skill <name>`,
`/skills`, or the `Skill` tool (`skill="list"` enumerates them). The index is
prefetched in the background so skills are available for fast lookup.

---

## Plugins

The plugin runtime (`plugins` crate) discovers, loads, and registers plugins from
`~/.mangocode/plugins/` and project `.mangocode/plugins/`. A plugin is a directory
with a `plugin.json` (or `plugin.toml`) manifest and content subdirectories:

```
my-plugin/
├── plugin.json          # manifest
├── commands/            # *.md slash commands
├── agents/              # *.md agent definitions
├── skills/              # SKILL.md folders
├── hooks/hooks.json     # lifecycle hooks
├── output-styles/       # *.md / *.json styles
└── .mcp.json            # MCP servers (optional)
```

A plugin can contribute **commands, agents, skills, output styles, MCP servers,
LSP servers, and hooks**. Manifests declare a `capabilities` list
(`read_files`, `write_files`, `network`, `shell`, `browser`, `mcp`) that is
enforced (an explicit empty list blocks everything; omitting it means
"old/trusted"). Manage plugins with `/plugin` (list/enable/disable/info/install/
reload) and `/reload-plugins`.

### Plugin hooks

Plugin `hooks.json` registers shell commands per `HookEventKind`
(`PreToolUse`, `PostToolUse`, `SessionStart`, `UserPromptSubmit`,
`SubagentStart/Stop`, `TaskCreated/Completed`, `FileChanged`, and many more), with
optional tool `matcher` globs and a `blocking` flag. Pre-tool / prompt-submit hooks
can deny an action; post-tool hooks are informational. Hooks run with
`CLAUDE_PLUGIN_ROOT` / `CLAUDE_PLUGIN_NAME` in the environment and receive the
event payload as JSON on stdin. See [crates/plugins.md](crates/plugins.md).

---

## MCP (Model Context Protocol)

The `mcp` crate is a full MCP client (protocol `2024-11-05`) over **stdio** and
**HTTP/SSE** transports. Configured servers (`Config.mcp_servers`, `--mcp-config`,
or plugin `.mcp.json`) are connected by a connection manager that performs the
initialize handshake, discovers tools/resources/prompts, and reconnects with
exponential backoff. Discovered tools are wrapped as `mcp__<server>__<tool>` and
exposed to the model alongside built-in tools.

MCP servers can be authenticated with OAuth (PKCE; tokens stored under
`~/.mangocode/mcp-tokens/`), and remote Pipedream MCP is supported with layered
token resolution. Environment variables in server configs are expanded
(`${VAR}` / `${VAR:-default}`). A static registry of well-known official servers
aids discovery. Browse connected servers and their tools with `/mcp`. See
[crates/mcp.md](crates/mcp.md).

---

## Coordination (multi-agent / multi-process)

`core::coordination` provides a local, network-free SQLite store (in
`~/.mangocode/sessions.db`) that lets independent MangoCode processes and sub-agents
see each other and avoid stepping on the same files:

- **Session presence** — heartbeated registry of active sessions (pid, cwd, repo,
  model, title).
- **Work claims** — claim a scope before editing; conflicting claims surface
  warnings.
- **Messages** — inbox/outbox between actors (`SendMessage`, `CoordinationInbox`).

The query loop's **coordinator mode** (`query::coordinator`) orchestrates worker
sub-agents: the coordinator is restricted to orchestration tools (`Agent`,
`SendMessage`, `TaskStop`, `TeamCreate`/`TeamDelete`, `StructuredOutput`) and may
delegate execution (e.g. `Bash`) to workers. The `Agent` tool can run workers in
isolated git worktrees or in the background. See
[crates/query.md](crates/query.md#sub-agents--coordination).
