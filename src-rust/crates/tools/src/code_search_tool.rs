use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct CodeSearchTool;

#[derive(Debug, Deserialize)]
struct CodeSearchInput {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    related_file: Option<String>,
    #[serde(default)]
    related_line: Option<usize>,
    #[serde(default)]
    include_text_files: bool,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    files: Vec<String>,
}

#[async_trait]
impl Tool for CodeSearchTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_CODE_SEARCH
    }

    fn description(&self) -> &str {
        "Search code by natural-language or symbol query using a chunked, identifier-aware local index. \
         Use this before Grep/Read when trying to locate where behavior is implemented. \
         Can also find chunks related to a file and line."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language or symbol/code query. Required unless related_file and related_line are provided."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to index and search. Defaults to the current working directory."
                },
                "top_k": {
                    "type": "number",
                    "description": "Number of chunks to return. Defaults to 5."
                },
                "related_file": {
                    "type": "string",
                    "description": "Optional file path for related-code search."
                },
                "related_line": {
                    "type": "number",
                    "description": "Line number in related_file for related-code search."
                },
                "include_text_files": {
                    "type": "boolean",
                    "description": "When true, also index text/document files such as Markdown, JSON, YAML, TOML, XML, and lock files."
                },
                "languages": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional language filters such as rust, python, typescript, or doc."
                },
                "files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional repo-relative file paths to restrict search results."
                }
            }
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: CodeSearchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let root = params
            .path
            .as_ref()
            .map(|path| ctx.resolve_path(path))
            .unwrap_or_else(|| ctx.working_dir.clone());
        if !root.is_dir() {
            return ToolResult::error(format!(
                "Search path is not a directory: {}",
                root.display()
            ));
        }

        let top_k = params.top_k.unwrap_or(5).clamp(1, 25);
        let mut index = match mangocode_file_search::FileSearchIndex::build_limited(&root, 20_000) {
            Ok(index) => index,
            Err(e) => {
                return ToolResult::error(format!("Failed to index {}: {}", root.display(), e));
            }
        };
        index.add_code_chunks_with_options(2_000, 512 * 1024, params.include_text_files);

        let query = params.query.as_deref().unwrap_or("").trim();
        let (hits, title) = if let (Some(file), Some(line)) =
            (params.related_file.as_deref(), params.related_line)
        {
            (
                index.find_related(file, line, top_k),
                format!("Code chunks related to {file}:{line}"),
            )
        } else {
            if query.is_empty() {
                return ToolResult::error(
                    "CodeSearch requires query, or both related_file and related_line.",
                );
            }
            (
                index.search_code_filtered(query, top_k, &params.languages, &params.files),
                format!("Code search results for {query:?}"),
            )
        };

        if hits.is_empty() {
            return ToolResult::success("No code search results found.");
        }

        let mut out = format!("{title}:\n");
        for (idx, hit) in hits.iter().enumerate() {
            out.push_str(&format!(
                "\n{}. {}:{}-{} (score {:.3})\n{}\n",
                idx + 1,
                hit.chunk.relative_path,
                hit.chunk.start_line,
                hit.chunk.end_line,
                hit.score,
                hit.chunk.content
            ));
        }
        ToolResult::success(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    fn test_context(working_dir: PathBuf) -> ToolContext {
        let handler = Arc::new(mangocode_core::permissions::AutoPermissionHandler {
            mode: mangocode_core::config::PermissionMode::Default,
        });
        ToolContext {
            working_dir,
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        }
    }

    #[tokio::test]
    async fn code_search_executes_query_and_related_without_query() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src").join("auth.rs"),
            "pub struct AuthManager {}\nfn validate_token() {}\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("src").join("session.rs"),
            "fn validate_session_token() {}\n",
        )
        .unwrap();

        let tool = CodeSearchTool;
        let ctx = test_context(dir.path().to_path_buf());

        let result = tool
            .execute(json!({ "query": "AuthManager", "top_k": 2 }), &ctx)
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("src/auth.rs"));

        let related = tool
            .execute(
                json!({ "related_file": "src/auth.rs", "related_line": 1, "top_k": 2 }),
                &ctx,
            )
            .await;
        assert!(!related.is_error);
        assert!(related.content.contains("src/session.rs"));
    }

    #[tokio::test]
    async fn code_search_rejects_empty_query_without_related_target() {
        let dir = tempfile::tempdir().unwrap();
        let tool = CodeSearchTool;
        let ctx = test_context(dir.path().to_path_buf());

        let result = tool.execute(json!({}), &ctx).await;

        assert!(result.is_error);
        assert!(result.content.contains("requires query"));
    }

    #[tokio::test]
    async fn code_search_supports_text_and_scope_filters() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::create_dir(dir.path().join("docs")).unwrap();
        std::fs::write(
            dir.path().join("src").join("auth.rs"),
            "fn validate_token() {}\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("docs").join("auth.md"),
            "Authentication runbook mentions validate_token behavior.\n",
        )
        .unwrap();

        let tool = CodeSearchTool;
        let ctx = test_context(dir.path().to_path_buf());

        let result = tool
            .execute(
                json!({
                    "query": "runbook validate token",
                    "include_text_files": true,
                    "languages": ["doc"],
                    "top_k": 3
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("docs/auth.md"));

        let file_scoped = tool
            .execute(
                json!({
                    "query": "validate token",
                    "include_text_files": true,
                    "files": ["src/auth.rs"],
                    "top_k": 3
                }),
                &ctx,
            )
            .await;
        assert!(!file_scoped.is_error);
        assert!(file_scoped.content.contains("src/auth.rs"));
        assert!(!file_scoped.content.contains("docs/auth.md"));
    }
}
