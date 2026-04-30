// WebFetch tool: HTTP GET with HTML-to-text conversion and native research-pipeline
// extraction, including optional rendered browser fallback for sparse docs pages.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, warn};

pub struct WebFetchTool;

#[derive(Debug, Deserialize)]
struct WebFetchInput {
    url: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    rendered_fallback: bool,
    #[serde(default = "default_extract_mode")]
    extract: String,
    #[serde(default = "default_citation_mode")]
    citation_mode: String,
    #[serde(default = "default_output_format")]
    output_format: String,
}

fn default_extract_mode() -> String {
    "auto".to_string()
}

fn default_citation_mode() -> String {
    "metadata".to_string()
}

fn default_output_format() -> String {
    "text".to_string()
}

/// Compute a simple hash of the URL and extraction options for cache purposes.
fn cache_key(params: &WebFetchInput) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    params.url.hash(&mut hasher);
    params.prompt.as_deref().unwrap_or("").hash(&mut hasher);
    params.rendered_fallback.hash(&mut hasher);
    params.extract.hash(&mut hasher);
    params.citation_mode.hash(&mut hasher);
    params.output_format.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Get the cache directory for web_fetch content.
fn get_cache_dir() -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".mangocode");
    dir.push("web_cache");
    dir
}

/// Attempt to load cached extracted content for a URL.
fn load_cached_extraction(params: &WebFetchInput) -> Option<String> {
    let cache_dir = get_cache_dir();
    let cache_file = cache_dir.join(format!("{}.txt", cache_key(params)));

    if cache_file.exists() {
        match fs::read_to_string(&cache_file) {
            Ok(content) => {
                debug!(file = ?cache_file, "Loaded cached web content");
                return Some(content);
            }
            Err(e) => {
                debug!(file = ?cache_file, error = %e, "Failed to load cache");
            }
        }
    }
    None
}

/// Save extracted content to cache.
fn save_cached_extraction(params: &WebFetchInput, content: &str) {
    let cache_dir = get_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        warn!(dir = ?cache_dir, error = %e, "Failed to create cache directory");
        return;
    }

    let cache_file = cache_dir.join(format!("{}.txt", cache_key(params)));
    if let Err(e) = fs::write(&cache_file, content) {
        warn!(file = ?cache_file, error = %e, "Failed to write cache file");
    } else {
        debug!(file = ?cache_file, "Cached extracted web content");
    }
}

fn prompt_filter_text(text: &str, prompt: &str) -> String {
    let prompt_terms: Vec<String> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| part.len() > 2)
        .map(|part| part.to_lowercase())
        .collect();

    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();

    if prompt_terms.is_empty() || paragraphs.is_empty() {
        return text.to_string();
    }

    let mut best_idx = 0usize;
    let mut best_score = 0usize;

    for (idx, paragraph) in paragraphs.iter().enumerate() {
        let lower = paragraph.to_lowercase();
        let score = prompt_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    if best_score == 0 {
        paragraphs
            .iter()
            .take(2)
            .copied()
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        paragraphs[best_idx].to_string()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_WEB_FETCH
    }

    fn description(&self) -> &str {
        "Fetches a web page URL and returns its content as text. HTML is \
         automatically converted to plain text. Use this for reading documentation, \
         APIs, and other web resources."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt for how to process the content"
                },
                "rendered_fallback": {
                    "type": "boolean",
                    "description": "Allow rendered browser fallback when HTTP extraction is sparse"
                },
                "extract": {
                    "type": "string",
                    "enum": ["auto", "main", "raw"],
                    "description": "Extraction mode (default auto)"
                },
                "citation_mode": {
                    "type": "string",
                    "enum": ["metadata", "inline", "none"],
                    "description": "Include retrieval metadata/citation header"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "description": "Preferred output format"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: WebFetchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Permission check
        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Fetch {}", params.url),
            true, // read-only
        ) {
            return ToolResult::error(e.to_string());
        }

        debug!(url = %params.url, "Fetching web page");

        // Load cached extracted content before any network work.
        if let Some(cached) = load_cached_extraction(&params) {
            return ToolResult::success(cached);
        }

        let (content_type, mut text, extraction_label) = if params.extract == "raw" {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::limited(10))
                .build();

            let client = match client {
                Ok(c) => c,
                Err(e) => return ToolResult::error(format!("Failed to create HTTP client: {}", e)),
            };

            let resp = match client
                .get(&params.url)
                .header("User-Agent", "MangoCode/1.0 web fetch")
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    return ToolResult::error(format!("Failed to fetch {}: {}", params.url, e));
                }
            };

            let status = resp.status();
            if !status.is_success() {
                return ToolResult::error(format!("HTTP {} when fetching {}", status, params.url));
            }

            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            let body = match resp.text().await {
                Ok(b) => b,
                Err(e) => return ToolResult::error(format!("Failed to read response body: {}", e)),
            };

            (content_type, body, "raw".to_string())
        } else {
            match crate::research::fetch_research_document(&params.url, params.rendered_fallback)
                .await
            {
                Ok(doc) => (
                    doc.content_type,
                    doc.content,
                    if doc.rendered_fallback_used {
                        "rendered".to_string()
                    } else {
                        params.extract.clone()
                    },
                ),
                Err(e) => return ToolResult::error(e),
            }
        };

        if let Some(prompt) = params.prompt.as_deref() {
            text = prompt_filter_text(&text, prompt);
        }

        // Truncate very long content
        const MAX_LEN: usize = 100_000;
        let mut text = if text.len() > MAX_LEN {
            format!(
                "{}\n\n... (truncated, {} total characters)",
                &text[..MAX_LEN],
                text.len()
            )
        } else {
            text
        };

        if params.output_format == "markdown" && !text.starts_with("# ") {
            text = format!("# WebFetch: {}\n\n{}", params.url, text);
        }

        if params.citation_mode != "none" {
            let header = format!(
                "Source: {}\nRetrieved: {}\nContent-Type: {}\nExtraction: {}\n\n",
                params.url,
                chrono::Utc::now().to_rfc3339(),
                content_type,
                extraction_label
            );
            text = format!("{}{}", header, text);
        }

        // Cache the final result
        save_cached_extraction(&params, &text);

        ToolResult::success(text)
    }
}
