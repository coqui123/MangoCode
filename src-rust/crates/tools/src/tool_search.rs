// ToolSearchTool: live tool discovery backed by the runtime registry.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct ToolSearchTool;

#[derive(Debug, Deserialize)]
struct ToolSearchInput {
    query: String,
    #[serde(default = "default_max")]
    max_results: usize,
}

fn default_max() -> usize {
    5
}

#[derive(Debug, Clone)]
struct LiveToolEntry {
    name: String,
    description: String,
    aliases: Vec<String>,
    source: String,
    search_terms: Vec<String>,
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "ToolSearch"
    }

    fn description(&self) -> &str {
        "Search for available tools by name, alias, source, or keyword. Use 'select:ToolName' \
         for direct lookup. Results are generated from the live registry, including built-ins, \
         aliases, and connected MCP tools."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query: use 'select:ToolName' for direct selection, or keywords to search"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum results to return (default: 5)"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ToolSearchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let query = params.query.trim();
        let max = params.max_results.clamp(1, 20);
        let catalog = live_tool_catalog(ctx);

        if let Some(names_str) = query.strip_prefix("select:").map(str::trim) {
            let requested = names_str
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty());
            let mut found = Vec::new();
            let mut missing = Vec::new();

            for name in requested {
                if let Some(entry) = catalog.iter().find(|entry| entry_name_matches(entry, name)) {
                    found.push(format_live_entry(entry));
                } else {
                    missing.push(name.to_string());
                }
            }

            if found.is_empty() {
                return ToolResult::success(format!(
                    "No matching tools found for: {}. Use a broader ToolSearch query or check whether the relevant plugin, MCP server, or feature flag is enabled.",
                    missing.join(", ")
                ));
            }

            let mut out = found.join("\n");
            if !missing.is_empty() {
                out.push_str(&format!("\n\nNot found: {}", missing.join(", ")));
            }
            return ToolResult::success(out);
        }

        let q_lower = query.to_lowercase();
        let terms: Vec<&str> = q_lower.split_whitespace().collect();

        let mut scored = catalog
            .iter()
            .filter_map(|entry| {
                let score = score_entry(entry, &terms);
                (score > 0).then_some((score, entry))
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        scored.truncate(max);

        if scored.is_empty() {
            return ToolResult::success(format!(
                "No tools found matching '{}'. Try broader keywords, an alias such as shell_command or apply_patch, or use 'select:ToolName'.",
                query
            ));
        }

        let lines = scored
            .iter()
            .map(|(_, entry)| format_live_entry(entry))
            .collect::<Vec<_>>();

        ToolResult::success(format!(
            "Tools matching '{}':\n\n{}\n\nTotal tools available: {}",
            query,
            lines.join("\n"),
            catalog.len()
        ))
    }
}

fn live_tool_catalog(ctx: &ToolContext) -> Vec<LiveToolEntry> {
    let tools = crate::all_tools();
    let mut entries = tools
        .iter()
        .map(|tool| LiveToolEntry {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            aliases: tool.aliases(),
            source: "built-in".to_string(),
            search_terms: Vec::new(),
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "tool-agent")]
    entries.push(LiveToolEntry {
        name: mangocode_core::constants::TOOL_NAME_AGENT.to_string(),
        description: "Launch a sub-agent for complex, multi-step work.".to_string(),
        aliases: crate::default_aliases_for_tool(mangocode_core::constants::TOOL_NAME_AGENT),
        source: "built-in:query".to_string(),
        search_terms: vec!["subagent".to_string(), "delegate".to_string()],
    });

    if let Some(manager) = ctx.mcp_manager.as_ref() {
        for (server_name, definition) in manager.all_tool_definitions() {
            let (name, aliases) = crate::mcp_compat_names(&server_name, &definition.name);
            entries.push(LiveToolEntry {
                name,
                description: definition.description,
                aliases,
                source: format!("mcp:{server_name}"),
                search_terms: vec![format!("mcp__{server_name}")],
            });
        }
    }

    entries.retain(|entry| entry_allowed_by_config(entry, ctx));
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

fn entry_name_matches(entry: &LiveToolEntry, requested: &str) -> bool {
    let requested = normalize_entry_name(requested);
    normalize_entry_name(&entry.name) == requested
        || entry
            .aliases
            .iter()
            .any(|alias| normalize_entry_name(alias) == requested)
}

fn entry_allowed_by_config(entry: &LiveToolEntry, ctx: &ToolContext) -> bool {
    let allowed = &ctx.config.allowed_tools;
    let disallowed = &ctx.config.disallowed_tools;
    (allowed.is_empty()
        || allowed
            .iter()
            .any(|requested| entry_name_matches(entry, requested)))
        && !disallowed
            .iter()
            .any(|requested| entry_name_matches(entry, requested))
}

fn normalize_entry_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

fn score_entry(entry: &LiveToolEntry, terms: &[&str]) -> usize {
    if terms.is_empty() {
        return 1;
    }

    let name_lower = entry.name.to_lowercase();
    let desc_lower = entry.description.to_lowercase();
    let source_lower = entry.source.to_lowercase();
    let mut score = 0usize;

    for term in terms {
        if name_lower == *term {
            score += 20;
        } else if name_lower.contains(term) {
            score += 10;
        }
        if desc_lower.contains(term) {
            score += 5;
        }
        if source_lower.contains(term) {
            score += 4;
        }
        for alias in &entry.aliases {
            let alias_lower = alias.to_lowercase();
            if alias_lower == *term {
                score += 12;
            } else if alias_lower.contains(term) {
                score += 6;
            }
        }
        for search_term in &entry.search_terms {
            let search_term_lower = search_term.to_lowercase();
            if search_term_lower == *term {
                score += 8;
            } else if search_term_lower.contains(term) {
                score += 4;
            }
        }
    }

    score
}

fn format_live_entry(entry: &LiveToolEntry) -> String {
    if entry.aliases.is_empty() {
        format!("{} [{}]: {}", entry.name, entry.source, entry.description)
    } else {
        format!(
            "{} [{}] aliases: {}\n  {}",
            entry.name,
            entry.source,
            entry.aliases.join(", "),
            entry.description
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(
        feature = "tool-agent",
        all(
            feature = "tool-apply-patch",
            feature = "tool-bash",
            feature = "tool-read"
        )
    ))]
    fn test_context() -> ToolContext {
        ToolContext {
            working_dir: std::path::PathBuf::from("/workspace"),
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: std::sync::Arc::new(
                mangocode_core::permissions::AutoPermissionHandler {
                    mode: mangocode_core::config::PermissionMode::Default,
                },
            ),
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "tool-search-test".to_string(),
            file_history: std::sync::Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        }
    }

    #[test]
    fn mcp_entries_use_executable_compat_name() {
        let (name, aliases) = crate::mcp_compat_names("github", "github_create_issue");
        assert_eq!(name, "mcp__github__create_issue");
        assert!(aliases.iter().any(|alias| alias == "github_create_issue"));
        assert!(aliases.iter().any(|alias| alias == "create_issue"));
        assert!(!aliases.iter().any(|alias| alias == "mcp__github"));
    }

    #[test]
    fn select_matches_aliases_case_insensitively() {
        let entry = LiveToolEntry {
            name: "mcp__github__create_issue".to_string(),
            description: String::new(),
            aliases: vec!["github_create_issue".to_string()],
            source: "mcp:github".to_string(),
            search_terms: Vec::new(),
        };
        assert!(entry_name_matches(&entry, "GITHUB_CREATE_ISSUE"));
    }

    #[test]
    fn select_matches_normalized_aliases() {
        let entry = LiveToolEntry {
            name: "Bash".to_string(),
            description: String::new(),
            aliases: vec!["shell_command".to_string(), "container.exec".to_string()],
            source: "built-in".to_string(),
            search_terms: Vec::new(),
        };

        assert!(entry_name_matches(&entry, "shell-command"));
        assert!(entry_name_matches(&entry, "container-exec"));
    }

    #[cfg(all(
        feature = "tool-apply-patch",
        feature = "tool-bash",
        feature = "tool-read"
    ))]
    #[test]
    fn live_catalog_respects_allowed_and_disallowed_tool_filters() {
        let mut ctx = test_context();
        ctx.config.allowed_tools = vec!["shell_command".to_string()];
        let catalog = live_tool_catalog(&ctx);
        assert!(catalog.iter().any(|entry| entry.name == "Bash"));
        assert!(!catalog.iter().any(|entry| entry.name == "Read"));

        let mut ctx = test_context();
        ctx.config.disallowed_tools = vec!["apply_patch".to_string()];
        let catalog = live_tool_catalog(&ctx);
        assert!(!catalog.iter().any(|entry| entry.name == "ApplyPatch"));
        assert!(catalog.iter().any(|entry| entry.name == "Read"));
    }

    #[cfg(feature = "tool-agent")]
    #[test]
    fn agent_is_in_live_catalog_as_query_builtin() {
        let ctx = test_context();
        let catalog = live_tool_catalog(&ctx);
        let agent = catalog
            .iter()
            .find(|entry| entry.name == mangocode_core::constants::TOOL_NAME_AGENT)
            .expect("Agent should be advertised in main query flows");
        assert!(agent.aliases.iter().any(|alias| alias == "spawn_agent"));
    }
}
