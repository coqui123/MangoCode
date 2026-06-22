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
    #[serde(default)]
    include_content: Option<bool>,
    #[serde(default)]
    max_snippet_chars: Option<usize>,
    #[serde(default)]
    semantic: Option<bool>,
}

#[async_trait]
impl Tool for CodeSearchTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_CODE_SEARCH
    }

    fn description(&self) -> &str {
        "Search code by natural-language or symbol query using a chunked, identifier-aware local index. \
         Use this before Grep/Read when trying to locate where behavior is implemented. \
         Can also find chunks related to a file and line. Use include_content=false or max_snippet_chars \
         when you only need compact path/line routing context."
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
                },
                "include_content": {
                    "type": "boolean",
                    "description": "When false, return only paths, line ranges, scores, and metadata. Defaults to true."
                },
                "max_snippet_chars": {
                    "type": "integer",
                    "description": "Optional maximum characters of content per hit to display. Defaults to full chunk content."
                },
                "semantic": {
                    "type": "boolean",
                    "description": "When true, re-rank keyword candidates by embedding similarity to the query (hybrid retrieval) using the local embedding model. Finds intent-matching code even when keywords differ. Slower; falls back to keyword ranking if no embedding backend is available. Ignored for related_file/related_line search."
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
        let include_content = params.include_content.unwrap_or(true);
        let max_snippet_chars = params.max_snippet_chars.map(|chars| chars.clamp(80, 8_000));
        let mut index = match mangocode_file_search::FileSearchIndex::build_limited(&root, 20_000) {
            Ok(index) => index,
            Err(e) => {
                return ToolResult::error(format!("Failed to index {}: {}", root.display(), e));
            }
        };
        index.add_code_chunks_with_options(2_000, 512 * 1024, params.include_text_files);

        let query = params.query.as_deref().unwrap_or("").trim();
        let semantic = params.semantic.unwrap_or(false);
        let (hits, title, semantic_used) = if let (Some(file), Some(line)) =
            (params.related_file.as_deref(), params.related_line)
        {
            (
                index.find_related(file, line, top_k),
                format!("Code chunks related to {file}:{line}"),
                false,
            )
        } else {
            if query.is_empty() {
                return ToolResult::error(
                    "CodeSearch requires query, or both related_file and related_line.",
                );
            }
            if semantic {
                // Hybrid retrieval: pull a wider keyword candidate set, then
                // re-rank by embedding similarity to the query. Embedding is
                // CPU/IO-bound, so run it off the async runtime.
                let candidate_n = (top_k * 6).clamp(top_k, 48);
                let candidates =
                    index.search_code_filtered(query, candidate_n, &params.languages, &params.files);
                let query_owned = query.to_string();
                let (reranked, used) =
                    tokio::task::spawn_blocking(move || semantic_rerank(&query_owned, candidates, top_k))
                        .await
                        .unwrap_or((Vec::new(), false));
                let title = if used {
                    format!("Semantic code search results for {query:?}")
                } else {
                    format!("Code search results for {query:?} (semantic unavailable; keyword ranking)")
                };
                (reranked, title, used)
            } else {
                (
                    index.search_code_filtered(query, top_k, &params.languages, &params.files),
                    format!("Code search results for {query:?}"),
                    false,
                )
            }
        };

        if hits.is_empty() {
            return ToolResult::success("No code search results found.").with_metadata(json!({
                "kind": "source_search",
                "source_paths": [],
                "relevant_files": [],
                "result_count": 0,
            }));
        }

        let source_paths = hits
            .iter()
            .map(|hit| hit.chunk.relative_path.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let mut out = format!("{title}:\n");
        for (idx, hit) in hits.iter().enumerate() {
            out.push_str(&format!(
                "\n{}. {}:{}-{} (score {:.3})\n",
                idx + 1,
                hit.chunk.relative_path,
                hit.chunk.start_line,
                hit.chunk.end_line,
                hit.score
            ));
            if include_content {
                out.push_str(&render_code_search_snippet(
                    &hit.chunk.content,
                    max_snippet_chars,
                ));
                out.push('\n');
            }
        }
        let relevant_files = source_paths.clone();
        ToolResult::success(out).with_metadata(json!({
            "kind": "source_search",
            "source_paths": source_paths,
            "relevant_files": relevant_files,
            "result_count": hits.len(),
            "content_included": include_content,
            "max_snippet_chars": max_snippet_chars,
            "semantic": semantic_used,
        }))
    }
}

/// Re-rank keyword candidates by embedding similarity to the query (hybrid
/// retrieval). Returns `(reranked_top_k, true)` when an embedding backend was
/// available, or `(keyword_top_k, false)` as a graceful fallback when no embedder
/// is configured. Embeds the query once, then each candidate's path+content
/// (bounded) as a passage and sorts by cosine similarity. Runs synchronously;
/// the caller wraps it in `spawn_blocking`.
fn semantic_rerank(
    query: &str,
    mut candidates: Vec<mangocode_file_search::CodeSearchHit>,
    top_k: usize,
) -> (Vec<mangocode_file_search::CodeSearchHit>, bool) {
    let Some(query_embedding) = mangocode_core::layered_memory::embed_text(query, true) else {
        candidates.truncate(top_k);
        return (candidates, false);
    };

    let mut scored: Vec<(f32, mangocode_file_search::CodeSearchHit)> = candidates
        .into_iter()
        .map(|hit| {
            // Bound the passage so a huge chunk doesn't dominate embedding cost.
            let mut passage = hit.chunk.relative_path.clone();
            passage.push('\n');
            passage.extend(hit.chunk.content.chars().take(2_000));
            let sim = mangocode_core::layered_memory::embed_text(&passage, false)
                .map(|emb| {
                    mangocode_core::layered_memory::embedding_cosine_similarity(
                        &query_embedding,
                        &emb,
                    )
                })
                .unwrap_or(f32::MIN);
            (sim, hit)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    let reranked = scored
        .into_iter()
        .map(|(sim, mut hit)| {
            // Surface the semantic score in the displayed ranking.
            hit.score = sim as f64;
            hit
        })
        .collect();
    (reranked, true)
}

fn render_code_search_snippet(content: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return content.to_string();
    };
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let mut snippet = content
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    snippet.push_str("...");
    snippet
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
            question_prompt_tx: None,
            cancel_token: None,
        }
    }

    fn make_hit(path: &str, content: &str, score: f64) -> mangocode_file_search::CodeSearchHit {
        mangocode_file_search::CodeSearchHit {
            chunk: mangocode_file_search::CodeChunk {
                content: content.to_string(),
                path: PathBuf::from(path),
                relative_path: path.to_string(),
                start_line: 1,
                end_line: 2,
                language: Some("rust".to_string()),
            },
            score,
        }
    }

    #[test]
    fn semantic_rerank_never_exceeds_top_k() {
        // Deterministic regardless of whether an embedding backend is present:
        // with one it re-ranks, without one it truncates — both bound to top_k.
        let candidates = vec![
            make_hit("a.rs", "fn login() {}", 3.0),
            make_hit("b.rs", "fn logout() {}", 2.0),
            make_hit("c.rs", "fn authenticate() {}", 1.0),
        ];
        let (out, _used) = semantic_rerank("user authentication", candidates, 2);
        assert!(out.len() <= 2, "rerank must never exceed top_k");
    }

    #[test]
    fn semantic_rerank_empty_candidates_is_safe() {
        let (out, used) = semantic_rerank("anything", Vec::new(), 5);
        assert!(out.is_empty());
        // No candidates means nothing to rank; flag reflects only that we produced
        // no semantic ordering, never panics.
        let _ = used;
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
        let source_paths = result
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("source_paths"))
            .and_then(serde_json::Value::as_array)
            .expect("source_paths metadata");
        assert!(source_paths
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|path| path == "src/auth.rs"));

        let related = tool
            .execute(
                json!({ "related_file": "src/auth.rs", "related_line": 1, "top_k": 2 }),
                &ctx,
            )
            .await;
        assert!(!related.is_error);
        assert!(related.content.contains("src/session.rs"));
        let related_paths = related
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("relevant_files"))
            .and_then(serde_json::Value::as_array)
            .expect("relevant_files metadata");
        assert!(related_paths
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|path| path == "src/session.rs"));
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

    #[tokio::test]
    async fn code_search_can_return_path_only_results_for_token_efficiency() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src").join("auth.rs"),
            "fn validate_token() {\n    let secret_padding = true;\n}\n",
        )
        .unwrap();

        let tool = CodeSearchTool;
        let ctx = test_context(dir.path().to_path_buf());
        let result = tool
            .execute(
                json!({
                    "query": "validate token",
                    "include_content": false,
                    "top_k": 1
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("src/auth.rs"));
        assert!(!result.content.contains("secret_padding"));
        assert_eq!(
            result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("content_included"))
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[tokio::test]
    async fn code_search_can_truncate_displayed_snippets_without_losing_hit_metadata() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src").join("auth.rs"),
            format!(
                "fn validate_token() {{}}\n{}\ntail_marker_should_not_render\n",
                "x".repeat(200)
            ),
        )
        .unwrap();

        let tool = CodeSearchTool;
        let ctx = test_context(dir.path().to_path_buf());
        let result = tool
            .execute(
                json!({
                    "query": "validate_token",
                    "max_snippet_chars": 80,
                    "top_k": 1
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("fn validate_token"));
        assert!(result.content.contains("..."));
        assert!(!result.content.contains("tail_marker_should_not_render"));
        assert_eq!(
            result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("max_snippet_chars"))
                .and_then(serde_json::Value::as_u64),
            Some(80)
        );
        let source_paths = result
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("source_paths"))
            .and_then(serde_json::Value::as_array)
            .expect("source_paths metadata");
        assert!(source_paths
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|path| path == "src/auth.rs"));
    }
}
