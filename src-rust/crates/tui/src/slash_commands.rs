// slash_commands.rs — Structured slash-command registry for prompt UX.

use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct SlashCommandSpec {
    pub name: String,
    pub description: String,
    pub group: String,
}

pub static PROMPT_SLASH_COMMANDS: Lazy<Vec<SlashCommandSpec>> = Lazy::new(|| {
    vec![
        SlashCommandSpec { name: "help".into(), description: "Show help".into(), group: "Core".into() },
        SlashCommandSpec { name: "clear".into(), description: "Clear the conversation transcript".into(), group: "Core".into() },
        SlashCommandSpec { name: "compact".into(), description: "Compact the conversation context".into(), group: "Core".into() },
        SlashCommandSpec { name: "context".into(), description: "Show context window and rate limit usage".into(), group: "Core".into() },
        SlashCommandSpec { name: "review".into(), description: "Review changes (git diff)".into(), group: "Core".into() },
        SlashCommandSpec { name: "ultrareview".into(), description: "Run an exhaustive multi-dimensional code review".into(), group: "Core".into() },
        SlashCommandSpec { name: "doctor".into(), description: "Run diagnostics".into(), group: "Core".into() },
        SlashCommandSpec { name: "init".into(), description: "Initialize AGENTS.md for this project".into(), group: "Core".into() },
        SlashCommandSpec { name: "insights".into(), description: "Generate a session analysis report with conversation statistics".into(), group: "Core".into() },
        SlashCommandSpec { name: "changes".into(), description: "Inspect changes from the current session".into(), group: "Core".into() },
        SlashCommandSpec { name: "diff".into(), description: "Inspect the current git diff".into(), group: "Core".into() },
        SlashCommandSpec { name: "rewind".into(), description: "Rewind to an earlier turn".into(), group: "Core".into() },

        SlashCommandSpec { name: "settings".into(), description: "Open settings".into(), group: "UI".into() },
        SlashCommandSpec { name: "theme".into(), description: "Open the theme picker".into(), group: "UI".into() },
        SlashCommandSpec { name: "privacy".into(), description: "Open privacy settings".into(), group: "UI".into() },
        SlashCommandSpec { name: "stats".into(), description: "Open token and cost stats".into(), group: "UI".into() },
        SlashCommandSpec { name: "survey".into(), description: "Open session feedback survey".into(), group: "UI".into() },
        SlashCommandSpec { name: "feedback".into(), description: "Open session feedback survey".into(), group: "UI".into() },
        SlashCommandSpec { name: "output-style".into(), description: "Toggle output style (auto/stream/verbose)".into(), group: "UI".into() },
        SlashCommandSpec { name: "copy".into(), description: "Copy the last assistant response to clipboard".into(), group: "UI".into() },

        SlashCommandSpec { name: "model".into(), description: "Change the AI model".into(), group: "Models".into() },
        SlashCommandSpec { name: "advisor".into(), description: "Set or unset the server-side advisor model".into(), group: "Models".into() },
        SlashCommandSpec { name: "effort".into(), description: "Set effort level (low/medium/high/max)".into(), group: "Models".into() },
        SlashCommandSpec { name: "fast".into(), description: "Toggle fast mode".into(), group: "Models".into() },

        SlashCommandSpec { name: "providers".into(), description: "List available AI providers and their status".into(), group: "Providers".into() },
        SlashCommandSpec { name: "connect".into(), description: "Connect an AI provider".into(), group: "Providers".into() },
        SlashCommandSpec { name: "login".into(), description: "Log in to MangoCode".into(), group: "Providers".into() },
        SlashCommandSpec { name: "logout".into(), description: "Log out of MangoCode".into(), group: "Providers".into() },
        SlashCommandSpec { name: "install-slack-app".into(), description: "Install the MangoCode Slack integration".into(), group: "Providers".into() },

        SlashCommandSpec { name: "session".into(), description: "Browse and manage sessions".into(), group: "Sessions".into() },
        SlashCommandSpec { name: "resume".into(), description: "Resume a previous session".into(), group: "Sessions".into() },
        SlashCommandSpec { name: "rename".into(), description: "Rename this session".into(), group: "Sessions".into() },
        SlashCommandSpec { name: "export".into(), description: "Export conversation".into(), group: "Sessions".into() },
        SlashCommandSpec { name: "fork".into(), description: "Fork session into a new branch".into(), group: "Sessions".into() },

        SlashCommandSpec { name: "agent".into(), description: "List available agents or show agent details".into(), group: "Agents".into() },
        SlashCommandSpec { name: "agents".into(), description: "Browse agent definitions and active agents".into(), group: "Agents".into() },
        SlashCommandSpec { name: "plugin".into(), description: "Manage plugins (list/info/enable/disable/reload)".into(), group: "Agents".into() },

        SlashCommandSpec { name: "mcp".into(), description: "Browse configured MCP servers".into(), group: "Integrations".into() },
        SlashCommandSpec { name: "hooks".into(), description: "Browse configured hooks (read-only)".into(), group: "Integrations".into() },
        SlashCommandSpec { name: "memory".into(), description: "Browse and open AGENTS.md memory files".into(), group: "Integrations".into() },

        SlashCommandSpec { name: "cost".into(), description: "Show cost breakdown".into(), group: "Diagnostics".into() },
        SlashCommandSpec { name: "keybindings".into(), description: "Show keybinding configuration".into(), group: "Diagnostics".into() },
        SlashCommandSpec { name: "heapdump".into(), description: "Show process memory and diagnostic information".into(), group: "Diagnostics".into() },
        SlashCommandSpec { name: "vim".into(), description: "Toggle vim keybindings".into(), group: "Diagnostics".into() },
        SlashCommandSpec { name: "voice".into(), description: "Toggle voice input mode".into(), group: "Diagnostics".into() },

        SlashCommandSpec { name: "quit".into(), description: "Quit MangoCode".into(), group: "System".into() },
        SlashCommandSpec { name: "exit".into(), description: "Quit MangoCode".into(), group: "System".into() },
        SlashCommandSpec { name: "config".into(), description: "Open settings".into(), group: "System".into() },
    ]
});

pub fn prompt_slash_commands(
    project_root: &std::path::Path,
    skills_config: &mangocode_core::config::SkillsConfig,
) -> Vec<SlashCommandSpec> {
    let mut commands = PROMPT_SLASH_COMMANDS.clone();
    let builtins: std::collections::HashSet<String> = commands.iter().map(|cmd| cmd.name.clone()).collect();

    for skill in mangocode_core::discover_skills(project_root, skills_config).into_values() {
        if builtins.contains(&skill.name) {
            continue;
        }

        commands.push(SlashCommandSpec {
            name: skill.name,
            description: skill.description,
            group: "Skills".into(),
        });
    }

    commands.sort_by(|a, b| a.group.cmp(&b.group).then(a.name.cmp(&b.name)));
    commands
}
