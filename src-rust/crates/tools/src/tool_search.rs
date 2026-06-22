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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolVisibility {
    Visible,
    Hidden(String),
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "ToolSearch"
    }

    fn description(&self) -> &str {
        "Search for runtime-visible tools by name, alias, source, or keyword. Use 'select:ToolName' \
         for direct lookup. Results are filtered by build features, session visibility config, and \
         connected MCP tools."
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
        let full_catalog = full_tool_catalog(ctx);
        let catalog = visible_tool_catalog_from(full_catalog.clone(), ctx);

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
                    missing.push(format_missing_tool(name, &full_catalog, ctx));
                }
            }

            if found.is_empty() {
                return ToolResult::success(format!(
                    "No matching runtime-visible tools found for: {}. Use a broader ToolSearch query or check whether the relevant plugin, MCP server, feature flag, and session visibility config permit it.",
                    missing.join("; ")
                ));
            }

            let mut out = found.join("\n");
            if !missing.is_empty() {
                out.push_str(&format!("\n\nNot found: {}", missing.join("; ")));
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
            let hidden_count = matching_hidden_tools(&full_catalog, ctx, &terms).len();
            let hidden_hint = if hidden_count == 0 {
                feature_gate_hint(query)
                    .map(|hint| format!(" {}", hint))
                    .unwrap_or_default()
            } else {
                // Report only the count — naming the restricted tools (or the
                // policy) would disclose the session's restriction surface.
                format!(
                    " {hidden_count} additional matching tool(s) are restricted by this session's configuration."
                )
            };
            return ToolResult::success(
                format!(
                "No runtime-visible tools found matching '{}'. Try broader keywords, use 'select:ToolName' with a visible tool name or alias, or check whether plugin, MCP, feature flag, and session visibility config allow the expected tool.",
                query
                ) + &hidden_hint,
            );
        }

        let lines = scored
            .iter()
            .map(|(_, entry)| format_live_entry(entry))
            .collect::<Vec<_>>();

        let hidden_count = matching_hidden_tools(&full_catalog, ctx, &terms).len();
        let hidden_hint = if hidden_count == 0 {
            String::new()
        } else {
            format!(
                "\n\n{hidden_count} additional matching tool(s) are restricted by this session's configuration."
            )
        };

        ToolResult::success(
            format!(
                "Tools matching '{}':\n\n{}\n\nTotal tools available: {}",
                query,
                lines.join("\n"),
                catalog.len()
            ) + &hidden_hint,
        )
    }
}

fn full_tool_catalog(ctx: &ToolContext) -> Vec<LiveToolEntry> {
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

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

#[cfg(test)]
fn live_tool_catalog(ctx: &ToolContext) -> Vec<LiveToolEntry> {
    visible_tool_catalog_from(full_tool_catalog(ctx), ctx)
}

fn visible_tool_catalog_from(entries: Vec<LiveToolEntry>, ctx: &ToolContext) -> Vec<LiveToolEntry> {
    entries
        .into_iter()
        .filter(|entry| entry_allowed_by_config(entry, ctx))
        .collect()
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
    matches!(entry_visibility(entry, ctx), ToolVisibility::Visible)
}

fn entry_visibility(entry: &LiveToolEntry, ctx: &ToolContext) -> ToolVisibility {
    let allowed = &ctx.config.allowed_tools;
    let disallowed = &ctx.config.disallowed_tools;
    if !allowed.is_empty()
        && !allowed
            .iter()
            .any(|requested| entry_name_matches(entry, requested))
    {
        // Do not enumerate the allowlist contents — in a locked-down session
        // that policy is a security boundary, not something to disclose to the
        // model (or to injected content driving it).
        return ToolVisibility::Hidden("hidden by session allowlist".to_string());
    }

    if disallowed
        .iter()
        .any(|requested| entry_name_matches(entry, requested))
    {
        return ToolVisibility::Hidden("hidden by session denylist".to_string());
    }

    ToolVisibility::Visible
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

fn matching_hidden_tools(
    entries: &[LiveToolEntry],
    ctx: &ToolContext,
    terms: &[&str],
) -> Vec<String> {
    let mut matches = entries
        .iter()
        .filter_map(|entry| {
            let score = score_entry(entry, terms);
            if score == 0 {
                return None;
            }
            match entry_visibility(entry, ctx) {
                ToolVisibility::Visible => None,
                ToolVisibility::Hidden(reason) => {
                    Some((score, format!("{} [{}]", entry.name, reason)))
                }
            }
        })
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    matches.into_iter().map(|(_, text)| text).collect()
}

fn format_missing_tool(name: &str, entries: &[LiveToolEntry], ctx: &ToolContext) -> String {
    if let Some(entry) = entries.iter().find(|entry| entry_name_matches(entry, name)) {
        return match entry_visibility(entry, ctx) {
            ToolVisibility::Visible => name.to_string(),
            ToolVisibility::Hidden(reason) => format!("{} ({})", entry.name, reason),
        };
    }

    feature_gate_hint(name)
        .map(|hint| format!("{name} ({hint})"))
        .unwrap_or_else(|| name.to_string())
}

fn feature_gate_hint(name: &str) -> Option<&'static str> {
    let name = normalize_entry_name(name);
    match name.as_str() {
        "browser" | "rendered_fetch" | "renderedfetch" => {
            Some("not built into this runtime unless the browser/rendered-fetch feature is enabled")
        }
        "computer" | "computer_use" | "computeruse" => {
            Some("not built into this runtime unless the computer-use feature is enabled")
        }
        name if name.starts_with("mcp__") => {
            Some("no connected MCP server currently exposes this runtime-visible tool")
        }
        _ => None,
    }
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
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: std::sync::Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
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

    #[tokio::test]
    async fn no_result_message_uses_runtime_visible_wording() {
        let ctx = test_context();
        let result = ToolSearchTool
            .execute(
                json!({
                    "query": "definitely-not-a-real-tool-name",
                    "max_results": 5
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("visible tool name or alias"));
        assert!(result.content.contains("session visibility config"));
        assert!(!result.content.contains("shell_command or apply_patch"));
    }

    #[cfg(feature = "tool-read")]
    #[tokio::test]
    async fn no_result_message_explains_hidden_allowlisted_tools() {
        let mut ctx = test_context();
        ctx.config.allowed_tools = vec!["ToolSearch".to_string()];

        let result = ToolSearchTool
            .execute(json!({ "query": "read", "max_results": 5 }), &ctx)
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("No runtime-visible tools found"));
        // The count is reported, but neither the hidden tool names nor the
        // allowlist contents are disclosed (session restriction surface).
        assert!(result
            .content
            .contains("restricted by this session's configuration"));
        assert!(!result.content.contains("Read ["));
        assert!(!result.content.contains("allowed tools are"));
    }

    #[tokio::test]
    async fn select_missing_message_mentions_visibility_config() {
        let ctx = test_context();
        let result = ToolSearchTool
            .execute(json!({ "query": "select:DefinitelyMissing" }), &ctx)
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("runtime-visible tools"));
        assert!(result.content.contains("session visibility config"));
    }

    #[cfg(feature = "tool-read")]
    #[tokio::test]
    async fn select_hidden_tool_reports_visibility_reason() {
        let mut ctx = test_context();
        ctx.config.disallowed_tools = vec!["read".to_string()];

        let result = ToolSearchTool
            .execute(json!({ "query": "select:Read" }), &ctx)
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("No matching runtime-visible tools"));
        // The user named the tool in the select: query, so echoing it back is
        // fine, but the denylist entry itself is not disclosed.
        assert!(result.content.contains("Read (hidden by session denylist)"));
        assert!(!result.content.contains("entry 'read'"));
    }

    #[cfg(not(feature = "tool-browser"))]
    #[tokio::test]
    async fn select_missing_browser_reports_feature_gate_hint() {
        let ctx = test_context();
        let result = ToolSearchTool
            .execute(json!({ "query": "select:Browser" }), &ctx)
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("browser/rendered-fetch feature"));
    }

    #[cfg(feature = "tool-grep")]
    #[tokio::test]
    async fn select_finds_code_search_tool() {
        let ctx = test_context();
        let result = ToolSearchTool
            .execute(json!({ "query": "select:CodeSearch" }), &ctx)
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("CodeSearch [built-in]"));
        assert!(result.content.contains("Search code by natural-language"));
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
