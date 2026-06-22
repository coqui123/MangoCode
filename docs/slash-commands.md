# Slash Commands

Slash commands are typed into the prompt with a leading `/` (e.g. `/model sonnet`).
They are implemented by the [`commands`](crates/commands.md) crate. This page is a
catalog of the built-in commands; the framework itself (the `SlashCommand` trait,
`CommandContext`, `CommandResult`) is documented in
[crates/commands.md](crates/commands.md).

How they run:

- The TUI parses input; `/`-prefixed input (but not `//`) is treated as a command.
- Some commands with arguments bypass the TUI overlay and are handled directly
  (e.g. `/model sonnet`, `/theme dark`).
- A command returns a `CommandResult` that drives the app: show a message, inject a
  user message, mutate config, clear/replace the conversation, open an overlay,
  start an OAuth flow, exit, etc.
- Definitions live in [`crates/commands/src/lib.rs`](../src-rust/crates/commands/src/lib.rs)
  and [`named_commands.rs`](../src-rust/crates/commands/src/named_commands.rs); the
  registry is `all_commands()`.

> The exact set depends on compiled features and may evolve. The registry in the
> source is authoritative; `/help` lists what's available in your build.

---

## Conversation

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/help` | `h`, `?` | List commands and usage. |
| `/clear` | `c`, `reset`, `new` | Clear the conversation. |
| `/compact` | | Compact the conversation to reduce tokens. |
| `/rewind` | | Open the message selector to restore an earlier state. |
| `/undo` | | Undo the last action / revert changes. |
| `/summary` | | Summarize the current conversation. |
| `/export` | | Export the conversation (JSON / Markdown). |
| `/fork` | | Create a new conversation branch. |
| `/branch` | | Create / list / switch conversation branches. |
| `/search` | | Search sessions and history. |
| `/copy` | | Copy text/objects to the clipboard. |

## Model, effort & output

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/model` | | Show or change the model. |
| `/effort` | | Show or set effort level. |
| `/fast` | | Toggle fast mode. |
| `/speed` | | Agent speed profile. |
| `/thinking` | | Configure extended thinking. |
| `/output-style` | | Switch output style. |
| `/providers` | | List / manage providers. |

## Settings & appearance

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/config` | `settings` | Show / modify configuration. |
| `/theme` | | Change theme. |
| `/color`, `/color-set` | | Prompt/accent colors. |
| `/keybindings` | | Open `~/.mangocode/keybindings.json`. |
| `/vim` | | Toggle vim mode. |
| `/privacy-settings` | | Privacy settings. |
| `/flags` | | List / toggle experimental feature flags. |
| `/statusline` | | Configure the status line. |
| `/voice` | | Toggle voice input/output. |
| `/sandbox-toggle` | | Toggle sandbox isolation. |
| `/rate-limit-options` | | Configure rate-limit handling. |

## Usage & cost

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/cost` | | Token usage & cost for this session. |
| `/usage` | | Current quota & usage. |
| `/extra-usage` | | Per-API-call token breakdown. |
| `/analytics` | | Session analytics / export event log. |
| `/stats` | | Detailed session statistics. |
| `/context` | | Context window usage & capacity. |
| `/ctx-viz` | | Visualize context composition. |

## System & status

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/status` | | System & session status. |
| `/version` | `v` | Version info. |
| `/doctor` | | Diagnose configuration issues. |
| `/run` | `work-run`, `workrun` | Inspect work-run lifecycle, evidence, replay trace. |
| `/upgrade` | | Check for / install updates. |
| `/release-notes` | | Release notes. |
| `/terminal-setup` | | Shell integration setup. |
| `/sleep` | | Pause for N milliseconds. |
| `/exit` | `quit`, `q` | Exit. |

## Auth & permissions

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/login` | | Authenticate (OAuth or API key). |
| `/logout` | | Clear credentials. |
| `/connect` | | Connect a provider/service. |
| `/permissions` | | Show / modify tool permission settings. |
| `/approvals-reviewer` | | Toggle auto-review vs manual approval. |
| `/critic` | | Toggle the permission critic. |
| `/completion-policy` | | Set completion policy (enforce/warn/off). |
| `/vault` | | Manage the credential vault (pre-session). |

## Project & code

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/workspace` | `cwd`, `cd`, `project` | Show / switch the active project directory. |
| `/add-dir` | | Add a directory to the workspace. |
| `/diff` | | Git diff of working-tree changes. |
| `/changes` | | Changes overview / diff viewer. |
| `/files` | | List project files. |
| `/memory` | | View / edit / review / clear project memory (`AGENTS.md`). |
| `/commit` | | Show staged changes and propose commits. |
| `/init` | | Initialize `AGENTS.md` / `.mangocode/`. |
| `/rename` | | Rename the session. |
| `/review` | | Request a code review. |
| `/security-review` | | Run security analysis. |
| `/bug` | | Report a bug / ask for debugging help. |
| `/intelligence` | `source-intelligence`, `intel` | Inspect/refresh source intelligence (ProjectGraph, CodeSearch). |
| `/graphify` | | ProjectGraph visualization (feature-gated). |
| `/insights` | | Deep session/code insights. |
| `/ultrareview` | | Deep multi-dimensional code review. |

## Planning & agents

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/plan` | | Enter planning mode / show the plan. |
| `/ultraplan` | | Agentic planner with extended thinking. |
| `/goal` | | View / create / update goals. |
| `/tasks` | | View / create / manage tasks. |
| `/agents` | | Manage and configure sub-agents. |
| `/proactive` | | Toggle the proactive agent. |
| `/think-back`, `/thinkback-play` | | Retroactive thinking / replay. |
| `/advisor` | | Get expert advice. |

## Integrations & extensions

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/mcp` | | Manage MCP servers. |
| `/hooks` | | View / manage hooks. |
| `/plugin` | | Manage plugins (list/enable/disable/info/install/reload). |
| `/reload-plugins` | | Reload all plugins. |
| `/skills`, `/skill` | | List / run skills; custom skill commands. |
| `/ide` | | Manage IDE integrations. |
| `/chrome` | | Chrome DevTools Protocol browser automation. |
| `/gateway` | | Configure API gateway routing. |
| `/pipedream` | | Pipedream MCP integration. |
| `/install-github-app` | | Set up GitHub Actions integration. |
| `/install-slack-app` | | Set up Slack integration. |
| `/pr-comments` | | Fetch GitHub PR comments. |

## Sessions & remote

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/resume` | `r`, `continue` | Resume a previous conversation. |
| `/session` | | View / list / manage saved sessions. |
| `/coordination` | `sessions` | Active local sessions & coordination state. |
| `/remote-control` | | Control remote MangoCode instances. |
| `/remote-env`, `/web-setup`, `/remote-setup` | | Configure the remote environment. |
| `/share` | | Share the session / create a link. |
| `/teleport` | | Move the session to another context. |
| `/desktop`, `/mobile` | | Desktop / mobile app links & deep links. |
| `/tag` | | Toggle searchable session tags. |
| `/passes`, `/stickers` | | Referral links / sticker page. |

## Diagnostics & misc

| Command | Aliases | Purpose |
| --- | --- | --- |
| `/feedback` | | Send feedback. |
| `/btw` | | Quick note utility. |
| `/heapdump` | | Capture/analyze memory usage. |

## User & plugin commands

Beyond the built-ins, slash commands can be defined by:

- **User templates** — `Config.commands` and markdown files under
  `~/.mangocode/commands/` or project `.mangocode/commands/`.
- **Plugins** — a plugin's `commands/*.md` (see [crates/plugins.md](crates/plugins.md)).
- **Skills** — `/skill <name>` runs a discovered skill (see
  [memory-skills-plugins.md](memory-skills-plugins.md)).
