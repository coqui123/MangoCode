// slash_commands.rs — Structured slash-command registry for prompt UX.

#[derive(Debug, Clone, Copy)]
pub struct SlashCommandSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub group: &'static str,
}

pub const PROMPT_SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec { name: "help", description: "Show help", group: "Core" },
    SlashCommandSpec { name: "clear", description: "Clear the conversation transcript", group: "Core" },
    SlashCommandSpec { name: "compact", description: "Compact the conversation context", group: "Core" },
    SlashCommandSpec { name: "context", description: "Show context window and rate limit usage", group: "Core" },
    SlashCommandSpec { name: "review", description: "Review changes (git diff)", group: "Core" },
    SlashCommandSpec { name: "ultrareview", description: "Run an exhaustive multi-dimensional code review", group: "Core" },
    SlashCommandSpec { name: "doctor", description: "Run diagnostics", group: "Core" },
    SlashCommandSpec { name: "init", description: "Initialize AGENTS.md for this project", group: "Core" },
    SlashCommandSpec { name: "insights", description: "Generate a session analysis report with conversation statistics", group: "Core" },
    SlashCommandSpec { name: "changes", description: "Inspect changes from the current session", group: "Core" },
    SlashCommandSpec { name: "diff", description: "Inspect the current git diff", group: "Core" },
    SlashCommandSpec { name: "rewind", description: "Rewind to an earlier turn", group: "Core" },

    SlashCommandSpec { name: "settings", description: "Open settings", group: "UI" },
    SlashCommandSpec { name: "theme", description: "Open the theme picker", group: "UI" },
    SlashCommandSpec { name: "privacy", description: "Open privacy settings", group: "UI" },
    SlashCommandSpec { name: "stats", description: "Open token and cost stats", group: "UI" },
    SlashCommandSpec { name: "survey", description: "Open session feedback survey", group: "UI" },
    SlashCommandSpec { name: "feedback", description: "Open session feedback survey", group: "UI" },
    SlashCommandSpec { name: "output-style", description: "Toggle output style (auto/stream/verbose)", group: "UI" },
    SlashCommandSpec { name: "copy", description: "Copy the last assistant response to clipboard", group: "UI" },

    SlashCommandSpec { name: "model", description: "Change the AI model", group: "Models" },
    SlashCommandSpec { name: "advisor", description: "Set or unset the server-side advisor model", group: "Models" },
    SlashCommandSpec { name: "effort", description: "Set effort level (low/medium/high/max)", group: "Models" },
    SlashCommandSpec { name: "fast", description: "Toggle fast mode", group: "Models" },

    SlashCommandSpec { name: "providers", description: "List available AI providers and their status", group: "Providers" },
    SlashCommandSpec { name: "connect", description: "Connect an AI provider", group: "Providers" },
    SlashCommandSpec { name: "login", description: "Log in to MangoCode", group: "Providers" },
    SlashCommandSpec { name: "logout", description: "Log out of MangoCode", group: "Providers" },
    SlashCommandSpec { name: "install-slack-app", description: "Install the MangoCode Slack integration", group: "Providers" },

    SlashCommandSpec { name: "session", description: "Browse and manage sessions", group: "Sessions" },
    SlashCommandSpec { name: "resume", description: "Resume a previous session", group: "Sessions" },
    SlashCommandSpec { name: "rename", description: "Rename this session", group: "Sessions" },
    SlashCommandSpec { name: "export", description: "Export conversation", group: "Sessions" },
    SlashCommandSpec { name: "fork", description: "Fork session into a new branch", group: "Sessions" },

    SlashCommandSpec { name: "agent", description: "List available agents or show agent details", group: "Agents" },
    SlashCommandSpec { name: "agents", description: "Browse agent definitions and active agents", group: "Agents" },
    SlashCommandSpec { name: "plugin", description: "Manage plugins (list/info/enable/disable/reload)", group: "Agents" },

    SlashCommandSpec { name: "mcp", description: "Browse configured MCP servers", group: "Integrations" },
    SlashCommandSpec { name: "hooks", description: "Browse configured hooks (read-only)", group: "Integrations" },
    SlashCommandSpec { name: "memory", description: "Browse and open AGENTS.md memory files", group: "Integrations" },

    SlashCommandSpec { name: "cost", description: "Show cost breakdown", group: "Diagnostics" },
    SlashCommandSpec { name: "keybindings", description: "Show keybinding configuration", group: "Diagnostics" },
    SlashCommandSpec { name: "heapdump", description: "Show process memory and diagnostic information", group: "Diagnostics" },
    SlashCommandSpec { name: "vim", description: "Toggle vim keybindings", group: "Diagnostics" },
    SlashCommandSpec { name: "voice", description: "Toggle voice input mode", group: "Diagnostics" },

    SlashCommandSpec { name: "quit", description: "Quit MangoCode", group: "System" },
    SlashCommandSpec { name: "exit", description: "Quit MangoCode", group: "System" },
    SlashCommandSpec { name: "config", description: "Open settings", group: "System" },
];
