// slash_commands.rs — Structured slash-command registry for prompt UX.

use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct SlashCommandSpec {
    pub name: String,
    pub description: String,
    pub group: String,
}

pub static PROMPT_SLASH_COMMANDS: Lazy<Vec<SlashCommandSpec>> = Lazy::new(|| {
    vec![
        SlashCommandSpec {
            name: "help".into(),
            description: "Show help".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "clear".into(),
            description: "Clear the conversation transcript".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "compact".into(),
            description: "Compact the conversation context".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "goal".into(),
            description: "Set or inspect the persistent session goal".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "context".into(),
            description: "Show context window and rate limit usage".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "review".into(),
            description: "Run a structured code review".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "ultrareview".into(),
            description: "Run an exhaustive multi-dimensional code review".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "graphify".into(),
            description:
                "Analyze, export graph/tree/callflow views, benchmark, inspect clusters, query, trace, and explain project architecture with ProjectGraph"
                    .into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "doctor".into(),
            description: "Run diagnostics".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "init".into(),
            description: "Initialize AGENTS.md for this project".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "insights".into(),
            description: "Generate a session analysis report with conversation statistics".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "coordination".into(),
            description: "Show active local sessions and claims".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "sessions".into(),
            description: "Alias for coordination status".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "changes".into(),
            description: "Inspect the latest file-changing turn".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "changes-export".into(),
            description: "Export latest turn changes as a patch bundle".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "diff".into(),
            description: "Inspect the current git diff".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "rewind".into(),
            description: "Rewind to an earlier turn".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "sleep".into(),
            description: "Toggle sleep prevention while turns run".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "approvals-reviewer".into(),
            description: "Toggle auto-review for approval decisions".into(),
            group: "Core".into(),
        },
        SlashCommandSpec {
            name: "settings".into(),
            description: "Open settings".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "theme".into(),
            description: "Open the theme picker".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "privacy".into(),
            description: "Open privacy settings".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "stats".into(),
            description: "Open token and cost stats".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "survey".into(),
            description: "Open session feedback survey".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "feedback".into(),
            description: "Open session feedback survey".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "output-style".into(),
            description: "Show or switch output style".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "copy".into(),
            description: "Copy the last assistant response to clipboard".into(),
            group: "UI".into(),
        },
        SlashCommandSpec {
            name: "model".into(),
            description: "Change the AI model".into(),
            group: "Models".into(),
        },
        SlashCommandSpec {
            name: "advisor".into(),
            description: "Set or unset the server-side advisor model".into(),
            group: "Models".into(),
        },
        SlashCommandSpec {
            name: "effort".into(),
            description: "Set effort level (low/medium/high/max)".into(),
            group: "Models".into(),
        },
        SlashCommandSpec {
            name: "fast".into(),
            description: "Toggle fast mode".into(),
            group: "Models".into(),
        },
        SlashCommandSpec {
            name: "providers".into(),
            description: "List available AI providers and their status".into(),
            group: "Providers".into(),
        },
        SlashCommandSpec {
            name: "connect".into(),
            description: "Connect an AI provider".into(),
            group: "Providers".into(),
        },
        SlashCommandSpec {
            name: "login".into(),
            description: "Log in to MangoCode".into(),
            group: "Providers".into(),
        },
        SlashCommandSpec {
            name: "logout".into(),
            description: "Log out of MangoCode".into(),
            group: "Providers".into(),
        },
        SlashCommandSpec {
            name: "install-slack-app".into(),
            description: "Install the MangoCode Slack integration".into(),
            group: "Providers".into(),
        },
        SlashCommandSpec {
            name: "session".into(),
            description: "Browse and manage sessions".into(),
            group: "Sessions".into(),
        },
        SlashCommandSpec {
            name: "resume".into(),
            description: "Resume a previous session".into(),
            group: "Sessions".into(),
        },
        SlashCommandSpec {
            name: "rename".into(),
            description: "Rename this session".into(),
            group: "Sessions".into(),
        },
        SlashCommandSpec {
            name: "export".into(),
            description: "Export conversation".into(),
            group: "Sessions".into(),
        },
        SlashCommandSpec {
            name: "fork".into(),
            description: "Fork session into a new branch".into(),
            group: "Sessions".into(),
        },
        SlashCommandSpec {
            name: "agent".into(),
            description: "List available agents or show agent details".into(),
            group: "Agents".into(),
        },
        SlashCommandSpec {
            name: "agents".into(),
            description: "Browse agent definitions and active agents".into(),
            group: "Agents".into(),
        },
        SlashCommandSpec {
            name: "plugin".into(),
            description: "Manage plugins (list/info/enable/disable/reload)".into(),
            group: "Agents".into(),
        },
        SlashCommandSpec {
            name: "mcp".into(),
            description: "Browse configured MCP servers".into(),
            group: "Integrations".into(),
        },
        SlashCommandSpec {
            name: "hooks".into(),
            description: "Browse configured hooks (read-only)".into(),
            group: "Integrations".into(),
        },
        SlashCommandSpec {
            name: "memory".into(),
            description: "Browse and open AGENTS.md memory files".into(),
            group: "Integrations".into(),
        },
        SlashCommandSpec {
            name: "cost".into(),
            description: "Show cost breakdown".into(),
            group: "Diagnostics".into(),
        },
        SlashCommandSpec {
            name: "keybindings".into(),
            description: "Open keybindings configuration".into(),
            group: "Diagnostics".into(),
        },
        SlashCommandSpec {
            name: "heapdump".into(),
            description: "Show process memory and diagnostic information".into(),
            group: "Diagnostics".into(),
        },
        SlashCommandSpec {
            name: "vim".into(),
            description: "Toggle vim keybindings".into(),
            group: "Diagnostics".into(),
        },
        SlashCommandSpec {
            name: "voice".into(),
            description: "Toggle voice input mode".into(),
            group: "Diagnostics".into(),
        },
        SlashCommandSpec {
            name: "quit".into(),
            description: "Quit MangoCode".into(),
            group: "System".into(),
        },
        SlashCommandSpec {
            name: "exit".into(),
            description: "Quit MangoCode".into(),
            group: "System".into(),
        },
        SlashCommandSpec {
            name: "config".into(),
            description: "Open settings".into(),
            group: "System".into(),
        },
    ]
});

// Keep this in sync with the runtime command registry in mangocode-commands.
// The commands crate has a test that fails if any runtime command name or alias
// is missing here. The TUI cannot import mangocode-commands without creating a
// dependency cycle, so this list is the explicit cross-crate contract.
pub const RUNTIME_SLASH_COMMAND_RESERVED_KEYS: &[&str] = &[
    "?",
    "add-dir",
    "advisor",
    "agent",
    "agents",
    "analytics",
    "approval-reviewer",
    "approvals-reviewer",
    "awake",
    "auto-review",
    "bashes",
    "branch",
    "btw",
    "bug",
    "c",
    "cd",
    "chrome",
    "clear",
    "color",
    "color-set",
    "commit",
    "compact",
    "config",
    "connect",
    "context",
    "context-visualizer",
    "continue",
    "coordination",
    "copy",
    "cost",
    "critic",
    "ctx",
    "ctx-viz",
    "cwd",
    "desktop",
    "diff",
    "doctor",
    "effort",
    "exit",
    "export",
    "extra-usage",
    "fast",
    "feedback",
    "find",
    "files",
    "flags",
    "fork",
    "gateway",
    "goal",
    "h",
    "heapdump",
    "help",
    "hooks",
    "ide",
    "init",
    "insights",
    "install-github-app",
    "install-slack-app",
    "keybindings",
    "login",
    "logout",
    "mcp",
    "memory",
    "mobile",
    "model",
    "new",
    "output-style",
    "passes",
    "permissions",
    "prevent-sleep",
    "pipedream",
    "plan",
    "plugin",
    "plugins",
    "privacy-settings",
    "proactive",
    "project",
    "providers",
    "pr-comments",
    "q",
    "quit",
    "r",
    "rate-limit-options",
    "rc",
    "release-notes",
    "reload-plugins",
    "remote",
    "remote-control",
    "remote-env",
    "remote-setup",
    "rename",
    "report",
    "reset",
    "resume",
    "review",
    "rewind",
    "sandbox",
    "sandbox-toggle",
    "search",
    "security-review",
    "session",
    "settings",
    "share",
    "sessions",
    "skill",
    "skills",
    "sleep",
    "speed",
    "stats",
    "status",
    "statusline",
    "stickers",
    "summary",
    "survey",
    "tag",
    "tasks",
    "teleport",
    "terminal-setup",
    "theme",
    "think",
    "think-back",
    "thinkback",
    "thinkback-play",
    "thinking",
    "undo",
    "update",
    "upgrade",
    "ultrareview",
    "usage",
    "v",
    "vault",
    "version",
    "vi",
    "vim",
    "voice",
    "web-setup",
    "workspace",
];

// Commands owned by the TUI layer rather than mangocode-commands. These still
// must be reserved so custom commands, skills, and plugins cannot run after the
// TUI opens a stateful screen for the same slash command.
pub const TUI_SLASH_COMMAND_RESERVED_KEYS: &[&str] = &["changes", "changes-export", "privacy"];

pub fn prompt_slash_commands(
    project_root: &std::path::Path,
    skills_config: &mangocode_core::config::SkillsConfig,
    command_templates: &HashMap<String, mangocode_core::CommandTemplate>,
) -> Vec<SlashCommandSpec> {
    let mut commands = PROMPT_SLASH_COMMANDS.clone();
    let mut seen = prompt_command_seen_keys(&commands);

    append_template_commands(&mut commands, &mut seen, command_templates);

    let skills_config = mangocode_plugins::skills_config_with_plugin_paths(skills_config);
    let mut skills: Vec<_> = mangocode_core::discover_skills(project_root, &skills_config)
        .into_values()
        .filter_map(|skill| {
            let name = normalized_skill_command_name(&skill.name)?;
            let normalized_name = name.to_string();
            Some((
                normalized_name.to_lowercase(),
                skill.name.trim().starts_with('/'),
                normalized_name,
                skill,
            ))
        })
        .collect();
    skills.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.source_path.cmp(&b.3.source_path))
    });

    for (lookup_key, _, name, skill) in skills {
        if !seen.insert(lookup_key) {
            continue;
        }

        commands.push(SlashCommandSpec {
            name,
            description: skill.description,
            group: "Skills".into(),
        });
    }

    if let Some(registry) = mangocode_plugins::global_plugin_registry() {
        append_plugin_commands(
            &mut commands,
            &mut seen,
            registry
                .all_command_defs()
                .into_iter()
                .map(|cmd| (cmd.name, cmd.description)),
        );
    }

    commands.sort_by(|a, b| a.group.cmp(&b.group).then(a.name.cmp(&b.name)));
    commands
}

fn prompt_command_seen_keys(commands: &[SlashCommandSpec]) -> HashSet<String> {
    let mut seen: HashSet<String> = commands.iter().map(|cmd| cmd.name.to_lowercase()).collect();
    seen.extend(
        RUNTIME_SLASH_COMMAND_RESERVED_KEYS
            .iter()
            .map(|key| key.to_lowercase()),
    );
    seen.extend(
        TUI_SLASH_COMMAND_RESERVED_KEYS
            .iter()
            .map(|key| key.to_lowercase()),
    );
    seen
}

fn append_template_commands(
    commands: &mut Vec<SlashCommandSpec>,
    seen: &mut HashSet<String>,
    templates: &HashMap<String, mangocode_core::CommandTemplate>,
) {
    let mut defs: Vec<(&str, &str, String)> = templates
        .iter()
        .filter_map(|(name, template)| {
            let normalized_name = normalized_template_command_name(name)?;
            let description = template
                .description
                .clone()
                .unwrap_or_else(|| "Custom command".to_string());
            Some((normalized_name, name.as_str(), description))
        })
        .collect();
    defs.sort_by(|a, b| {
        a.0.to_lowercase()
            .cmp(&b.0.to_lowercase())
            .then(
                a.1.trim()
                    .starts_with('/')
                    .cmp(&b.1.trim().starts_with('/')),
            )
            .then(a.0.cmp(b.0))
            .then(a.1.cmp(b.1))
    });

    for (name, _, description) in defs {
        if !seen.insert(name.to_lowercase()) {
            continue;
        }

        commands.push(SlashCommandSpec {
            name: name.to_string(),
            description,
            group: "Custom".into(),
        });
    }
}

fn normalized_template_command_name(name: &str) -> Option<&str> {
    let name = name.trim().trim_start_matches('/').trim();
    (!name.is_empty()).then_some(name)
}

fn normalized_skill_command_name(name: &str) -> Option<&str> {
    let name = normalized_template_command_name(name)?;
    let name = strip_markdown_suffix(name).trim();
    (!name.is_empty()).then_some(name)
}

fn strip_markdown_suffix(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() >= 3 && bytes[bytes.len() - 3..].eq_ignore_ascii_case(b".md") {
        &name[..name.len() - 3]
    } else {
        name
    }
}

fn append_plugin_commands<I>(
    commands: &mut Vec<SlashCommandSpec>,
    seen: &mut HashSet<String>,
    defs: I,
) where
    I: IntoIterator<Item = (String, String)>,
{
    let mut defs: Vec<_> = defs
        .into_iter()
        .filter_map(|(name, description)| {
            let normalized_name = normalized_template_command_name(&name)?.to_string();
            Some((
                normalized_name.to_lowercase(),
                name.trim().starts_with('/'),
                normalized_name,
                name,
                description,
            ))
        })
        .collect();
    defs.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(a.4.cmp(&b.4))
    });

    for (lookup_key, _, name, _, description) in defs {
        if !seen.insert(lookup_key) {
            continue;
        }

        commands.push(SlashCommandSpec {
            name,
            description,
            group: "Plugins".into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_plugin_commands_skips_existing_names() {
        let mut commands = vec![SlashCommandSpec {
            name: "help".into(),
            description: "Show help".into(),
            group: "Core".into(),
        }];
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                ("plugin-build".to_string(), "Build from plugin".to_string()),
                ("help".to_string(), "Shadow help".to_string()),
            ],
        );

        assert_eq!(commands.len(), 2);
        assert!(commands.iter().any(|cmd| cmd.name == "plugin-build"));
        assert_eq!(commands.iter().filter(|cmd| cmd.name == "help").count(), 1);
    }

    #[test]
    fn append_template_commands_adds_custom_commands_and_skips_core_collisions() {
        let mut commands = vec![SlashCommandSpec {
            name: "help".into(),
            description: "Show help".into(),
            group: "Core".into(),
        }];
        let mut seen = prompt_command_seen_keys(&commands);
        let mut templates = HashMap::new();
        templates.insert(
            "Ship".to_string(),
            mangocode_core::CommandTemplate {
                template: "Ship $ARGUMENTS".to_string(),
                description: Some("Ship the change".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "Help".to_string(),
            mangocode_core::CommandTemplate {
                template: "Shadow help".to_string(),
                description: Some("Shadow help".to_string()),
                ..Default::default()
            },
        );

        append_template_commands(&mut commands, &mut seen, &templates);

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "Ship" && cmd.group == "Custom"));
        assert_eq!(commands.iter().filter(|cmd| cmd.name == "help").count(), 1);
        assert!(!commands.iter().any(|cmd| cmd.name == "Help"));
    }

    #[test]
    fn append_template_commands_skips_runtime_reserved_names_and_aliases() {
        let mut commands = PROMPT_SLASH_COMMANDS.clone();
        let mut seen = prompt_command_seen_keys(&commands);
        let mut templates = HashMap::new();
        templates.insert(
            "Ship".to_string(),
            mangocode_core::CommandTemplate {
                template: "Ship $ARGUMENTS".to_string(),
                description: Some("Ship the change".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "h".to_string(),
            mangocode_core::CommandTemplate {
                template: "Shadow help alias".to_string(),
                description: Some("Shadow help alias".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "Analytics".to_string(),
            mangocode_core::CommandTemplate {
                template: "Shadow runtime-only command".to_string(),
                description: Some("Shadow analytics".to_string()),
                ..Default::default()
            },
        );

        append_template_commands(&mut commands, &mut seen, &templates);

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "Ship" && cmd.group == "Custom"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "h" && cmd.group == "Custom"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "Analytics" && cmd.group == "Custom"));
    }

    #[test]
    fn append_template_commands_prefers_unslashed_duplicate_keys() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);
        let mut templates = HashMap::new();
        templates.insert(
            "/Build".to_string(),
            mangocode_core::CommandTemplate {
                template: "Slash build".to_string(),
                description: Some("Slash build".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "Build".to_string(),
            mangocode_core::CommandTemplate {
                template: "Plain build".to_string(),
                description: Some("Plain build".to_string()),
                ..Default::default()
            },
        );

        append_template_commands(&mut commands, &mut seen, &templates);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "Build");
        assert_eq!(commands[0].description, "Plain build");
    }

    #[test]
    fn append_template_commands_prefers_unslashed_duplicate_with_spacing_and_case() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);
        let mut templates = HashMap::new();
        templates.insert(
            " /Build ".to_string(),
            mangocode_core::CommandTemplate {
                template: "Slash build".to_string(),
                description: Some("Slash build".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "build".to_string(),
            mangocode_core::CommandTemplate {
                template: "Plain build".to_string(),
                description: Some("Plain build".to_string()),
                ..Default::default()
            },
        );

        append_template_commands(&mut commands, &mut seen, &templates);

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "build");
        assert_eq!(commands[0].description, "Plain build");
    }

    #[test]
    fn append_template_commands_preserves_markdown_suffix() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);
        let mut templates = HashMap::new();
        templates.insert(
            "Build".to_string(),
            mangocode_core::CommandTemplate {
                template: "Plain build".to_string(),
                description: Some("Plain build".to_string()),
                ..Default::default()
            },
        );
        templates.insert(
            "Build.md".to_string(),
            mangocode_core::CommandTemplate {
                template: "Markdown build".to_string(),
                description: Some("Markdown build".to_string()),
                ..Default::default()
            },
        );

        append_template_commands(&mut commands, &mut seen, &templates);

        assert!(commands.iter().any(|cmd| cmd.name == "Build"));
        assert!(commands.iter().any(|cmd| cmd.name == "Build.md"));
    }

    #[test]
    fn append_plugin_commands_skips_case_insensitive_existing_names() {
        let mut commands = vec![SlashCommandSpec {
            name: "help".into(),
            description: "Show help".into(),
            group: "Core".into(),
        }];
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [("Help".to_string(), "Shadow help".to_string())],
        );

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "help");
    }

    #[test]
    fn append_plugin_commands_skips_runtime_reserved_names_and_aliases() {
        let mut commands = PROMPT_SLASH_COMMANDS.clone();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                ("h".to_string(), "Shadow help alias".to_string()),
                ("Analytics".to_string(), "Shadow analytics".to_string()),
                ("plugin-build".to_string(), "Build from plugin".to_string()),
            ],
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "plugin-build" && cmd.group == "Plugins"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "h" && cmd.group == "Plugins"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "Analytics" && cmd.group == "Plugins"));
    }

    #[test]
    fn append_plugin_commands_skips_tui_reserved_names() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                ("changes".to_string(), "Shadow changes".to_string()),
                ("plugin-build".to_string(), "Build from plugin".to_string()),
            ],
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "plugin-build" && cmd.group == "Plugins"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "changes" && cmd.group == "Plugins"));
    }

    #[test]
    fn append_plugin_commands_normalizes_display_names() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                (
                    " /plugin-build ".to_string(),
                    "Build from plugin".to_string(),
                ),
                ("   ".to_string(), "Blank plugin command".to_string()),
            ],
        );

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "plugin-build");
        assert_eq!(commands[0].group, "Plugins");
    }

    #[test]
    fn append_plugin_commands_prefers_unslashed_duplicate_names() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                (
                    " /plugin-build ".to_string(),
                    "Slash plugin build".to_string(),
                ),
                ("plugin-build".to_string(), "Plain plugin build".to_string()),
            ],
        );

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "plugin-build");
        assert_eq!(commands[0].description, "Plain plugin build");
    }

    #[test]
    fn append_plugin_commands_preserves_markdown_suffix() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [
                ("plugin-build".to_string(), "Plain plugin build".to_string()),
                (
                    "plugin-build.md".to_string(),
                    "Markdown plugin build".to_string(),
                ),
            ],
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "plugin-build" && cmd.group == "Plugins"));
        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "plugin-build.md" && cmd.group == "Plugins"));
    }

    #[test]
    fn append_plugin_commands_preserves_nested_colon_names() {
        let mut commands = Vec::new();
        let mut seen = prompt_command_seen_keys(&commands);

        append_plugin_commands(
            &mut commands,
            &mut seen,
            [(
                "toolbox:build:deploy".to_string(),
                "Deploy nested build command".to_string(),
            )],
        );

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "toolbox:build:deploy");
        assert_eq!(commands[0].group, "Plugins");
    }

    #[test]
    fn prompt_slash_commands_skips_case_insensitive_skill_collisions() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("help.md"),
            "---\nname: Help\ndescription: Shadow help\n---\nShadow.",
        )
        .unwrap();

        let commands = prompt_slash_commands(
            tmp.path(),
            &mangocode_core::config::SkillsConfig::default(),
            &HashMap::new(),
        );

        assert_eq!(commands.iter().filter(|cmd| cmd.name == "help").count(), 1);
        assert!(!commands.iter().any(|cmd| cmd.name == "Help"));
    }

    #[test]
    fn prompt_slash_commands_skips_runtime_reserved_skill_names_and_aliases() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("skill.md"),
            "---\nname: skill\ndescription: Shadow skills alias\n---\nShadow.",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("analytics.md"),
            "---\nname: analytics\ndescription: Shadow runtime command\n---\nShadow.",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("ship.md"),
            "---\nname: ship\ndescription: Ship skill\n---\nShip.",
        )
        .unwrap();

        let commands = prompt_slash_commands(
            tmp.path(),
            &mangocode_core::config::SkillsConfig::default(),
            &HashMap::new(),
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "ship" && cmd.group == "Skills"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "skill" && cmd.group == "Skills"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "analytics" && cmd.group == "Skills"));
    }

    #[test]
    fn prompt_slash_commands_normalizes_skill_names() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("ship.md"),
            "---\nname: /Ship.MD\ndescription: Ship skill\n---\nShip.",
        )
        .unwrap();

        let commands = prompt_slash_commands(
            tmp.path(),
            &mangocode_core::config::SkillsConfig::default(),
            &HashMap::new(),
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "Ship" && cmd.group == "Skills"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "/Ship" && cmd.group == "Skills"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "Ship.MD" && cmd.group == "Skills"));
    }

    #[test]
    fn prompt_slash_commands_dedupes_normalized_skill_names() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("plain.md"),
            "---\nname: Ship\ndescription: Plain ship\n---\nShip.",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("slash.md"),
            "---\nname: /Ship\ndescription: Slash ship\n---\nShip.",
        )
        .unwrap();

        let commands = prompt_slash_commands(
            tmp.path(),
            &mangocode_core::config::SkillsConfig::default(),
            &HashMap::new(),
        );

        let matches = commands
            .iter()
            .filter(|cmd| cmd.name == "Ship" && cmd.group == "Skills")
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].description, "Plain ship");
    }

    #[test]
    fn prompt_slash_commands_lists_custom_before_skill_collisions() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().join(".mangocode").join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("ship.md"),
            "---\nname: ship\ndescription: Skill ship\n---\nShip skill.",
        )
        .unwrap();
        let mut templates = HashMap::new();
        templates.insert(
            "Ship".to_string(),
            mangocode_core::CommandTemplate {
                template: "Ship template".to_string(),
                description: Some("Template ship".to_string()),
                ..Default::default()
            },
        );

        let commands = prompt_slash_commands(
            tmp.path(),
            &mangocode_core::config::SkillsConfig::default(),
            &templates,
        );

        assert!(commands
            .iter()
            .any(|cmd| cmd.name == "Ship" && cmd.group == "Custom"));
        assert!(!commands
            .iter()
            .any(|cmd| cmd.name == "ship" && cmd.group == "Skills"));
    }
}
