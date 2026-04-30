//! Coding-oriented document research tools.
//!
//! These tools provide the first native layer of Mango Research. Rendered
//! fallback poaches the adaptive-browser pattern as a native HTTP +
//! script-aware extraction pass without requiring users to install sidecars.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const MAX_DOC_CHARS: usize = 80_000;
const RESEARCH_CACHE_VERSION: &str = "research-v1";
const RESEARCH_CACHE_TTL_SECS: u64 = 60 * 60 * 24;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchDocument {
    pub url: String,
    pub title: Option<String>,
    pub content: String,
    pub content_type: String,
    pub retrieved_at: String,
    pub quality_score: f32,
    pub rendered_fallback_used: bool,
    #[serde(default)]
    pub verified_at_read_time: bool,
    #[serde(default)]
    pub from_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResearchIndexEntry {
    url: String,
    title: Option<String>,
    content_excerpt: String,
    content_type: String,
    retrieved_at: String,
    quality_score: f32,
    rendered_fallback_used: bool,
    terms: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DocSearchInput {
    query: String,
    #[serde(default = "default_max_sources")]
    max_results: usize,
    #[serde(default = "default_source_preference")]
    source_preference: String,
    #[serde(default)]
    site_filters: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DocReadInput {
    url: String,
    #[serde(default)]
    purpose: Option<String>,
    #[serde(default)]
    rendered_fallback: bool,
    #[serde(default = "default_citation_mode")]
    citation_mode: String,
}

#[derive(Debug, Deserialize)]
struct DeepReadInput {
    query: String,
    #[serde(default)]
    urls: Vec<String>,
    #[serde(default = "default_max_sources")]
    max_sources: usize,
    #[serde(default)]
    purpose: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RenderedFetchInput {
    url: String,
    #[serde(default = "default_citation_mode")]
    citation_mode: String,
    #[serde(default = "default_research_output_format")]
    output_format: String,
}

fn default_citation_mode() -> String {
    "inline".to_string()
}

fn default_research_output_format() -> String {
    "markdown".to_string()
}

fn default_max_sources() -> usize {
    5
}

pub struct DocReadTool;
pub struct DeepReadTool;
pub struct DocSearchTool;
pub struct RenderedFetchTool;

fn default_source_preference() -> String {
    "official".to_string()
}

#[async_trait]
impl Tool for DocSearchTool {
    fn name(&self) -> &str {
        "DocSearch"
    }

    fn description(&self) -> &str {
        "Search for documentation and primary sources, preferring official docs, repositories, release notes, and standards."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Docs or API search query" },
                "max_results": { "type": "number", "description": "Maximum results to return (default 5, max 10)" },
                "source_preference": { "type": "string", "enum": ["official", "primary", "any"], "description": "How aggressively to rank official/primary sources first" },
                "site_filters": { "type": "array", "items": { "type": "string" }, "description": "Optional domains to restrict search, e.g. docs.rs" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: DocSearchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid DocSearch input: {}", e)),
        };
        if let Err(e) =
            ctx.check_permission(self.name(), &format!("Search docs {}", params.query), true)
        {
            return ToolResult::error(e.to_string());
        }

        let site_filters = merged_site_filters(&params.query, &params.site_filters);
        let query = apply_site_filters(&params.query, &params.site_filters);
        let mut urls = search_research_index(&query, params.max_results.clamp(1, 10))
            .into_iter()
            .map(|entry| entry.url)
            .collect::<Vec<_>>();
        if site_filters.is_empty() {
            urls.extend(discover_local_docs(
                &ctx.working_dir,
                &params.query,
                params.max_results,
            ));
        }
        let mut search_warning = None;
        let web_budget = params.max_results.clamp(1, 10).saturating_sub(urls.len());
        match discover_urls(&query, web_budget.max(1)).await {
            Ok(mut web_urls) => urls.append(&mut web_urls),
            Err(e) => {
                if urls.is_empty() {
                    return ToolResult::error(e);
                }
                search_warning = Some(e);
            }
        }
        urls.dedup();
        urls.retain(|url| url_matches_site_filters(url, &site_filters));
        if params.source_preference != "any" {
            urls.sort_by_key(|u| official_source_rank(u));
        }
        urls.truncate(params.max_results.clamp(1, 10));
        let mut results = urls
            .into_iter()
            .enumerate()
            .map(|(idx, url)| format!("{}. {}{}", idx + 1, url, source_badge(&url)))
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(warning) = search_warning {
            results.push_str(&format!(
                "\n\n[web search unavailable; returned local docs only: {}]",
                warning
            ));
        }
        ToolResult::success(if results.is_empty() {
            "No documentation results found.".to_string()
        } else {
            results
        })
    }
}

#[async_trait]
impl Tool for DocReadTool {
    fn name(&self) -> &str {
        "DocRead"
    }

    fn description(&self) -> &str {
        "Read a documentation URL or web page into clean, citation-ready Markdown for coding tasks. Uses HTTP first and reports when rendered browser fallback would be needed."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "URL to read" },
                "purpose": { "type": "string", "description": "What coding decision this source should inform" },
                "rendered_fallback": { "type": "boolean", "description": "Allow rendered browser fallback when HTTP content is sparse" },
                "citation_mode": { "type": "string", "enum": ["inline", "metadata", "none"], "description": "How to include source citation metadata" }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: DocReadInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid DocRead input: {}", e)),
        };
        if let Err(e) =
            ctx.check_permission(self.name(), &format!("Read docs {}", params.url), true)
        {
            return ToolResult::error(e.to_string());
        }
        let read_result = if looks_like_local_doc_ref(&params.url) {
            let path = resolve_local_doc_ref(ctx, &params.url);
            read_local_research_document(&path)
        } else {
            fetch_research_document(&params.url, params.rendered_fallback).await
        };
        match read_result {
            Ok(doc) => ToolResult::success(format_doc_read(
                &doc,
                params.purpose.as_deref(),
                &params.citation_mode,
            )),
            Err(e) => ToolResult::error(e),
        }
    }
}

#[async_trait]
impl Tool for DeepReadTool {
    fn name(&self) -> &str {
        "DeepRead"
    }

    fn description(&self) -> &str {
        "Read multiple sources and return a concise research brief for coding: summary, facts, source list, conflicts, examples, and implementation implications."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Research question or docs topic" },
                "urls": { "type": "array", "items": { "type": "string" }, "description": "Known source URLs to read before searching" },
                "max_sources": { "type": "number", "description": "Maximum sources to read (default 5, max 8)" },
                "purpose": { "type": "string", "description": "What implementation this research should support" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: DeepReadInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid DeepRead input: {}", e)),
        };
        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Deep research {}", params.query),
            true,
        ) {
            return ToolResult::error(e.to_string());
        }

        let max_sources = params.max_sources.clamp(1, 8);
        let urls = if params.urls.is_empty() {
            let mut urls = discover_local_docs(&ctx.working_dir, &params.query, max_sources);
            if urls.len() < max_sources {
                let mut web_urls = discover_urls(&params.query, max_sources)
                    .await
                    .unwrap_or_default();
                urls.append(&mut web_urls);
                urls.dedup();
                urls.sort_by_key(|u| official_source_rank(u));
                urls.truncate(max_sources);
            }
            urls
        } else {
            params.urls.into_iter().take(max_sources).collect()
        };

        if urls.is_empty() {
            return ToolResult::error("No URLs were provided or discovered for DeepRead.");
        }

        let mut docs = Vec::new();
        let mut failures = Vec::new();
        for url in urls {
            let read_result = if looks_like_local_doc_ref(&url) {
                let path = resolve_local_doc_ref(ctx, &url);
                read_local_research_document(&path)
            } else {
                fetch_research_document(&url, true).await
            };
            match read_result {
                Ok(doc) => docs.push(doc),
                Err(e) => failures.push(format!("- {}: {}", url, e)),
            }
        }

        if docs.is_empty() {
            return ToolResult::error(format!(
                "DeepRead could not read any sources.\n{}",
                failures.join("\n")
            ));
        }

        ToolResult::success(format_research_brief(
            &params.query,
            params.purpose.as_deref(),
            &docs,
            &failures,
        ))
    }
}

#[async_trait]
impl Tool for RenderedFetchTool {
    fn name(&self) -> &str {
        "RenderedFetch"
    }

    fn description(&self) -> &str {
        "Fetch a page through the browser-backed rendered extraction path and return citation-ready text or Markdown."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "URL to render and extract" },
                "citation_mode": { "type": "string", "enum": ["inline", "metadata", "none"], "description": "How to include source citation metadata" },
                "output_format": { "type": "string", "enum": ["text", "markdown"], "description": "Preferred output format" }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: RenderedFetchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid RenderedFetch input: {}", e)),
        };
        if let Err(e) =
            ctx.check_permission(self.name(), &format!("Rendered fetch {}", params.url), true)
        {
            return ToolResult::error(e.to_string());
        }

        match crate::browser_tool::rendered_extract_for_research(&params.url).await {
            Ok(content) => {
                let doc = ResearchDocument {
                    url: params.url.clone(),
                    title: None,
                    quality_score: score_content_quality(&content),
                    content_type: "text/html".to_string(),
                    retrieved_at: chrono::Utc::now().to_rfc3339(),
                    rendered_fallback_used: true,
                    verified_at_read_time: true,
                    from_cache: false,
                    content,
                };
                let _ = save_cached_document(&doc);
                let _ = upsert_research_index(&doc);
                ToolResult::success(format_rendered_fetch(
                    &doc.url,
                    &doc.content,
                    &params.citation_mode,
                    &params.output_format,
                ))
            }
            Err(e) => ToolResult::error(format!("RenderedFetch failed for {}: {}", params.url, e)),
        }
    }
}

fn looks_like_local_doc_ref(value: &str) -> bool {
    value.starts_with("file://")
        || (!value.starts_with("http://")
            && !value.starts_with("https://")
            && (value.contains('/') || value.contains('\\') || Path::new(value).exists()))
}

fn resolve_local_doc_ref(ctx: &ToolContext, value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("file://") {
        PathBuf::from(rest.trim_start_matches('/'))
    } else {
        ctx.resolve_path(value)
    }
}

fn read_local_research_document(path: &Path) -> Result<ResearchDocument, String> {
    if !path.exists() {
        return Err(format!("Local document not found: {}", path.display()));
    }
    if path.is_dir() {
        return Err(format!("Local document is a directory: {}", path.display()));
    }
    let content = match mangocode_core::smart_attachments::classify_path(path) {
        mangocode_core::smart_attachments::AttachmentKind::Pdf
        | mangocode_core::smart_attachments::AttachmentKind::OfficeDocument
        | mangocode_core::smart_attachments::AttachmentKind::Html
        | mangocode_core::smart_attachments::AttachmentKind::Data
        | mangocode_core::smart_attachments::AttachmentKind::Archive => {
            mangocode_core::smart_attachments::extract_markdown_native(path)
                .map(|extracted| extracted.markdown)
                .map_err(|e| format!("Local document extraction failed: {}", e))?
        }
        _ => std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read local document: {}", e))?,
    };
    let content_type = mangocode_core::smart_attachments::media_type_for_path(path)
        .unwrap_or_else(|| "text/plain".to_string());
    let quality_score = score_content_quality(&content);
    let doc = ResearchDocument {
        url: format!("file://{}", path.display()),
        title: path
            .file_name()
            .and_then(|s| s.to_str())
            .map(str::to_string),
        content,
        content_type,
        retrieved_at: chrono::Utc::now().to_rfc3339(),
        quality_score,
        rendered_fallback_used: false,
        verified_at_read_time: true,
        from_cache: false,
    };
    let _ = upsert_research_index(&doc);
    Ok(doc)
}

pub async fn fetch_research_document(
    url: &str,
    rendered_fallback: bool,
) -> Result<ResearchDocument, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    if let Some(mut cached) = load_cached_document(url) {
        cached.verified_at_read_time = verify_source_reachable(&client, url).await;
        if cached.verified_at_read_time {
            if rendered_fallback && should_upgrade_cached_doc(&cached) {
                if let Some(upgraded) =
                    upgrade_document_with_rendered_fallback(cached.clone()).await
                {
                    let _ = save_cached_document(&upgraded);
                    let _ = upsert_research_index(&upgraded);
                    return Ok(upgraded);
                }
            }
            let _ = upsert_research_index(&cached);
            return Ok(cached);
        }
    }

    let resp = match send_research_get(&client, url).await {
        Ok(resp) => resp,
        Err(e) if is_docs_rs_url(url) => return Ok(docs_rs_fallback_document(url, &e)),
        Err(e) => return Err(e),
    };

    let status = resp.status();
    if !status.is_success() {
        if is_docs_rs_url(url) {
            return Ok(docs_rs_fallback_document(
                url,
                &format!("HTTP {} when fetching {}", status, url),
            ));
        }
        return Err(format!("HTTP {} when fetching {}", status, url));
    }
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {}", e))?;
    let title = extract_title(&body);
    let mut content = if content_type.contains("html") {
        html_to_markdownish(&body)
    } else {
        body
    };
    let mut quality_score = score_content_quality(&content);
    let mut rendered_fallback_used = false;
    if rendered_fallback && quality_score < 0.35 {
        let rendered = match crate::browser_tool::rendered_extract_for_research(url).await {
            Ok(markdown) => markdown,
            Err(_) => native_rendered_fallback_extract(&content),
        };
        let rendered_quality = score_content_quality(&rendered);
        if rendered_quality > quality_score {
            content = rendered;
            quality_score = score_content_quality(&content);
            rendered_fallback_used = true;
        } else {
            content.push_str(
                "\n\n[Rendered fallback attempted, but the extracted content remains sparse.]",
            );
        }
    }
    if content.len() > MAX_DOC_CHARS {
        content = format!(
            "{}\n\n... (truncated, {} total characters)",
            mangocode_core::truncate::truncate_bytes_prefix(&content, MAX_DOC_CHARS),
            content.len()
        );
    }

    let doc = ResearchDocument {
        url: url.to_string(),
        title,
        content,
        content_type,
        retrieved_at: chrono::Utc::now().to_rfc3339(),
        quality_score,
        rendered_fallback_used,
        verified_at_read_time: true,
        from_cache: false,
    };
    let _ = save_cached_document(&doc);
    let _ = upsert_research_index(&doc);
    Ok(doc)
}

async fn verify_source_reachable(client: &reqwest::Client, url: &str) -> bool {
    match client
        .head(url)
        .header("User-Agent", "MangoCode/1.0 docs research")
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => true,
        Ok(resp) if resp.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED => client
            .get(url)
            .header("User-Agent", "MangoCode/1.0 docs research")
            .send()
            .await
            .map(|resp| resp.status().is_success())
            .unwrap_or(false),
        Ok(_) => false,
        Err(_) => false,
    }
}

async fn send_research_get(
    client: &reqwest::Client,
    url: &str,
) -> Result<reqwest::Response, String> {
    match perform_research_get(client, url).await {
        Ok(resp) => Ok(resp),
        Err(first_error) => {
            let retry_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::limited(10))
                .http1_only()
                .build()
                .map_err(|e| format!("Failed to create HTTP retry client: {}", e))?;
            perform_research_get(&retry_client, url)
                .await
                .map_err(|retry_error| {
                    format!(
                        "Failed to fetch {}: {} (HTTP/1.1 retry also failed: {})",
                        url, first_error, retry_error
                    )
                })
        }
    }
}

async fn perform_research_get(
    client: &reqwest::Client,
    url: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    client
        .get(url)
        .header("User-Agent", "MangoCode/1.0 docs research")
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        )
        .send()
        .await
}

fn should_upgrade_cached_doc(doc: &ResearchDocument) -> bool {
    !doc.rendered_fallback_used && doc.quality_score < 0.35
}

fn is_docs_rs_url(url: &str) -> bool {
    url.to_ascii_lowercase().contains("docs.rs/")
}

fn docs_rs_fallback_document(url: &str, reason: &str) -> ResearchDocument {
    let content = format!(
        "# docs.rs fetch fallback\n\nSource attempted: {}\n\nMangoCode could not fetch this exact docs.rs page: {}\n\n{}\n",
        url,
        reason,
        docs_rs_fallback_links(url)
            .into_iter()
            .map(|(label, link)| format!("- {}: {}", label, link))
            .collect::<Vec<_>>()
            .join("\n")
    );
    ResearchDocument {
        url: url.to_string(),
        title: Some("docs.rs fetch fallback".to_string()),
        content,
        content_type: "text/markdown".to_string(),
        retrieved_at: chrono::Utc::now().to_rfc3339(),
        quality_score: 0.25,
        rendered_fallback_used: false,
        verified_at_read_time: false,
        from_cache: false,
    }
}

fn docs_rs_fallback_links(url: &str) -> Vec<(&'static str, &'static str)> {
    let lower = url.to_ascii_lowercase();
    if lower.contains("tokio") {
        vec![
            (
                "Tokio graceful shutdown guide",
                "https://tokio.rs/tokio/topics/shutdown",
            ),
            (
                "tokio::task::JoinHandle",
                "https://docs.rs/tokio/latest/tokio/task/struct.JoinHandle.html",
            ),
            (
                "tokio::task::AbortHandle",
                "https://docs.rs/tokio/latest/tokio/task/struct.AbortHandle.html",
            ),
            (
                "tokio_util::sync::CancellationToken",
                "https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html",
            ),
            (
                "tokio_util::task::TaskTracker",
                "https://docs.rs/tokio-util/latest/tokio_util/task/task_tracker/struct.TaskTracker.html",
            ),
        ]
    } else {
        vec![("docs.rs crate root", "https://docs.rs/")]
    }
}

async fn upgrade_document_with_rendered_fallback(
    mut doc: ResearchDocument,
) -> Option<ResearchDocument> {
    let rendered = match crate::browser_tool::rendered_extract_for_research(&doc.url).await {
        Ok(markdown) => markdown,
        Err(_) => native_rendered_fallback_extract(&doc.content),
    };
    let rendered_quality = score_content_quality(&rendered);
    if rendered_quality <= doc.quality_score {
        return None;
    }
    doc.content = rendered;
    doc.quality_score = rendered_quality;
    doc.rendered_fallback_used = true;
    doc.verified_at_read_time = true;
    doc.from_cache = false;
    doc.retrieved_at = chrono::Utc::now().to_rfc3339();
    Some(doc)
}

fn native_rendered_fallback_extract(content: &str) -> String {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.contains("__next_data__")
            || lower.contains("application/ld+json")
            || lower.contains("window.__")
            || lower.contains("props")
            || lower.contains("markdown")
            || lower.contains("code")
            || lower.contains("example")
        {
            out.push(decode_escaped_text(trimmed));
        }
    }
    if out.is_empty() {
        content.to_string()
    } else {
        out.join("\n")
    }
}

fn decode_escaped_text(text: &str) -> String {
    text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\/", "/")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

async fn discover_urls(query: &str, max_sources: usize) -> Result<Vec<String>, String> {
    let mut urls = Vec::new();
    urls.extend(curated_official_candidates(query));

    let instant_url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        encode_query(query)
    );
    if let Ok(resp) = reqwest::get(&instant_url).await {
        if let Ok(data) = resp.json::<Value>().await {
            if let Some(url) = data.get("AbstractURL").and_then(|v| v.as_str()) {
                if !url.is_empty() {
                    urls.push(url.to_string());
                }
            }
            collect_related_urls(data.get("RelatedTopics"), &mut urls, max_sources * 2);
        }
    }

    if urls.len() < max_sources {
        if let Ok(mut html_urls) = discover_duckduckgo_html_urls(query, max_sources * 3).await {
            urls.append(&mut html_urls);
        }
    }

    if urls.is_empty() {
        return Err("Search did not return any usable documentation URLs".to_string());
    }
    urls.sort_by_key(|u| official_source_rank(u));
    urls.dedup();
    urls.truncate(max_sources);
    Ok(urls)
}

async fn discover_duckduckgo_html_urls(
    query: &str,
    max_sources: usize,
) -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("Search client failed: {}", e))?;
    for search_url in [
        format!("https://duckduckgo.com/html/?q={}", encode_query(query)),
        format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            encode_query(query)
        ),
    ] {
        let html = client
            .get(search_url)
            .header("User-Agent", "MangoCode/1.0 docs research")
            .send()
            .await
            .map_err(|e| format!("HTML search failed: {}", e))?
            .text()
            .await
            .map_err(|e| format!("HTML search body failed: {}", e))?;
        let urls = extract_search_result_urls(&html, max_sources);
        if !urls.is_empty() {
            return Ok(urls);
        }
    }
    Ok(Vec::new())
}

fn curated_official_candidates(query: &str) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    let terms = query_terms(query);
    let mut urls = Vec::new();
    if lower.contains("tokio") {
        urls.push("https://tokio.rs/tokio/topics/shutdown".to_string());
        urls.push("https://docs.rs/tokio/latest/tokio/task/struct.JoinHandle.html".to_string());
        urls.push("https://docs.rs/tokio/latest/tokio/task/struct.AbortHandle.html".to_string());
        urls.push(
            "https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html"
                .to_string(),
        );
        urls.push(
            "https://docs.rs/tokio-util/latest/tokio_util/task/task_tracker/struct.TaskTracker.html"
                .to_string(),
        );
    }
    for term in terms.iter().take(4) {
        if lower.contains("rust") || lower.contains("crate") || lower.contains("docs.rs") {
            urls.push(format!(
                "https://docs.rs/{}/latest/{}",
                term,
                term.replace('-', "_")
            ));
            urls.push(format!("https://crates.io/crates/{}", term));
        }
        if lower.contains("npm")
            || lower.contains("node")
            || lower.contains("typescript")
            || lower.contains("javascript")
        {
            urls.push(format!("https://www.npmjs.com/package/{}", term));
            urls.push(format!(
                "https://github.com/search?q={}+documentation&type=repositories",
                term
            ));
        }
        if lower.contains("python") || lower.contains("pip") || lower.contains("pypi") {
            urls.push(format!("https://pypi.org/project/{}/", term));
            urls.push(format!("https://{}.readthedocs.io/", term));
        }
    }
    urls
}

fn extract_search_result_urls(html: &str, max_sources: usize) -> Vec<String> {
    let mut urls = Vec::new();
    let mut cursor = 0usize;
    while let Some(idx) = html[cursor..].find("uddg=") {
        let start = cursor + idx + "uddg=".len();
        let end = html[start..]
            .find(['&', '"', '\''])
            .map(|rel| start + rel)
            .unwrap_or(html.len());
        let raw = &html[start..end];
        let decoded = percent_decode(raw);
        if decoded.starts_with("http://") || decoded.starts_with("https://") {
            urls.push(decoded);
        }
        cursor = end;
        if urls.len() >= max_sources {
            break;
        }
    }
    if urls.is_empty() {
        let mut cursor = 0usize;
        while let Some(idx) = html[cursor..].find("href=\"http") {
            let start = cursor + idx + "href=\"".len();
            let end = html[start..]
                .find('"')
                .map(|rel| start + rel)
                .unwrap_or(html.len());
            urls.push(decode_entities(&html[start..end]));
            cursor = end;
            if urls.len() >= max_sources {
                break;
            }
        }
    }
    urls.retain(|url| !url.contains("duckduckgo.com"));
    urls.sort_by_key(|u| official_source_rank(u));
    urls.dedup();
    urls.truncate(max_sources);
    urls
}

fn discover_local_docs(root: &Path, query: &str, max_sources: usize) -> Vec<String> {
    let terms = query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|term| term.len() >= 3)
        .map(str::to_string)
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Vec::new();
    }

    let mut stack = vec![root.to_path_buf()];
    let mut scored = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if name == ".git" || name == "target" || name == "node_modules" || name == ".mangocode"
            {
                continue;
            }
            if path.is_dir() {
                if stack.len() < 200 {
                    stack.push(path);
                }
                continue;
            }
            if !is_doc_candidate(&path) {
                continue;
            }
            if should_skip_local_doc_candidate(&path, &terms) {
                continue;
            }
            let path_text = path.display().to_string().to_ascii_lowercase();
            let name_score = terms
                .iter()
                .filter(|term| name.contains(term.as_str()) || path_text.contains(term.as_str()))
                .count();
            let content_score = if name_score == 0 {
                std::fs::read_to_string(&path)
                    .ok()
                    .map(|content| {
                        let lower = content.to_ascii_lowercase();
                        terms
                            .iter()
                            .filter(|term| lower.contains(term.as_str()))
                            .count()
                    })
                    .unwrap_or(0)
            } else {
                0
            };
            let score = name_score * 3 + content_score;
            if score >= 2 {
                scored.push((score, format!("file://{}", path.display())));
            }
            if scored.len() > max_sources.saturating_mul(8).max(32) {
                break;
            }
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(max_sources)
        .map(|(_, path)| path)
        .collect()
}

fn is_doc_candidate(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some(
            "md" | "markdown"
                | "txt"
                | "rst"
                | "html"
                | "htm"
                | "pdf"
                | "docx"
                | "pptx"
                | "xlsx"
                | "json"
                | "xml"
                | "csv"
                | "toml"
                | "yaml"
                | "yml"
        )
    )
}

fn should_skip_local_doc_candidate(path: &Path, terms: &[String]) -> bool {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let manifest_like = matches!(
        file_name.as_str(),
        "cargo.toml"
            | "package.json"
            | "tsconfig.json"
            | "pyproject.toml"
            | "composer.json"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
    );
    if !manifest_like {
        return false;
    }

    let manifest_terms = [
        "cargo",
        "manifest",
        "dependency",
        "dependencies",
        "package",
        "workspace",
        "toml",
        "config",
        "configuration",
        "gradle",
        "pom",
    ];
    !terms
        .iter()
        .any(|term| manifest_terms.iter().any(|needle| term == needle))
}

fn collect_related_urls(value: Option<&Value>, urls: &mut Vec<String>, max_sources: usize) {
    let Some(Value::Array(items)) = value else {
        return;
    };
    for item in items {
        if urls.len() >= max_sources {
            break;
        }
        if let Some(url) = item.get("FirstURL").and_then(|v| v.as_str()) {
            if !url.is_empty() {
                urls.push(url.to_string());
            }
        }
        collect_related_urls(item.get("Topics"), urls, max_sources);
    }
}

fn format_doc_read(doc: &ResearchDocument, purpose: Option<&str>, citation_mode: &str) -> String {
    let mut out = String::new();
    if citation_mode != "none" {
        out.push_str(&format!(
            "Source: {}\nTitle: {}\nRetrieved: {}\nVerification: {}\nContent-Type: {}\nQuality: {:.2}\nCache: {}\n",
            doc.url,
            doc.title.as_deref().unwrap_or("(untitled)"),
            doc.retrieved_at,
            verification_status(doc.verified_at_read_time),
            doc.content_type,
            doc.quality_score,
            if doc.from_cache { "hit" } else { "miss" }
        ));
        if let Some(purpose) = purpose {
            out.push_str(&format!("Purpose: {}\n", purpose));
        }
        out.push('\n');
    }
    out.push_str(&doc.content);
    out
}

fn format_rendered_fetch(
    url: &str,
    content: &str,
    citation_mode: &str,
    output_format: &str,
) -> String {
    let mut out = String::new();
    if citation_mode != "none" {
        out.push_str(&format!(
            "Source: {}\nRetrieved: {}\nVerification: {}\nExtraction: rendered-browser\n\n",
            url,
            chrono::Utc::now().to_rfc3339(),
            verification_status(true)
        ));
    }
    if output_format == "markdown" && !content.starts_with("# ") {
        out.push_str(&format!("# RenderedFetch: {}\n\n", url));
    }
    out.push_str(content);
    out
}

fn format_research_brief(
    query: &str,
    purpose: Option<&str>,
    docs: &[ResearchDocument],
    failures: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("# Research Brief\n\n");
    out.push_str(&format!("Query: {}\n", query));
    if let Some(purpose) = purpose {
        out.push_str(&format!("Purpose: {}\n", purpose));
    }
    out.push('\n');

    out.push_str("## Answer Summary\n");
    out.push_str("Use the source excerpts below to ground implementation decisions. Prefer the official or primary sources listed first.\n\n");

    out.push_str("## Relevant Facts\n");
    for (idx, doc) in docs.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} ({})\n",
            idx + 1,
            doc.title.as_deref().unwrap_or("(untitled)"),
            doc.url
        ));
        out.push_str(&excerpt(&doc.content, 1200));
        out.push_str("\n\n");
    }

    out.push_str("## Source List\n");
    for (idx, doc) in docs.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} - retrieved {} - verification {} - quality {:.2}\n",
            idx + 1,
            doc.url,
            doc.retrieved_at,
            verification_status(doc.verified_at_read_time),
            doc.quality_score
        ));
    }
    if !failures.is_empty() {
        out.push_str("\n## Unreadable Sources\n");
        out.push_str(&failures.join("\n"));
        out.push('\n');
    }

    out.push_str("\n## Conflict/Staleness Notes\n");
    out.push_str(&conflict_and_staleness_notes(docs));
    out.push('\n');
    out.push_str("## Code/API Examples\n");
    let examples = collect_code_examples(docs);
    if examples.is_empty() {
        out.push_str("No compact code/API examples were detected in the extracted source text.\n");
    } else {
        out.push_str(&examples.join("\n\n"));
        out.push('\n');
    }
    out.push_str("\n## Implementation Implications\n");
    out.push_str(&implementation_implications(docs));
    out
}

fn html_to_markdownish(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let lower_bytes = lower.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if !in_tag && bytes[i] == b'<' {
            in_tag = true;
            let rest = &lower_bytes[i..lower_bytes.len().min(i + 20)];
            if rest.starts_with(b"<script") {
                in_script = true;
            } else if rest.starts_with(b"</script") {
                in_script = false;
            } else if rest.starts_with(b"<style") {
                in_style = true;
            } else if rest.starts_with(b"</style") {
                in_style = false;
            } else if rest.starts_with(b"<pre")
                || rest.starts_with(b"<code")
                || rest.starts_with(b"<p")
                || rest.starts_with(b"<li")
                || rest.starts_with(b"<h")
                || rest.starts_with(b"<br")
                || rest.starts_with(b"<tr")
            {
                text.push('\n');
            }
            i += 1;
            continue;
        }
        if in_tag {
            if bytes[i] == b'>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }
        if !in_script && !in_style {
            text.push(bytes[i] as char);
        }
        i += 1;
    }
    decode_entities(&text)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title>")? + "<title>".len();
    let end = lower[start..].find("</title>")? + start;
    Some(decode_entities(html[start..end].trim()))
}

fn score_content_quality(content: &str) -> f32 {
    let words = content.split_whitespace().count() as f32;
    let code_markers = [
        "```",
        "function ",
        "class ",
        "const ",
        "let ",
        "fn ",
        "pub ",
        "import ",
    ]
    .iter()
    .filter(|marker| content.contains(**marker))
    .count() as f32;
    ((words / 500.0).min(0.75) + (code_markers * 0.05).min(0.25)).min(1.0)
}

fn official_source_rank(url: &str) -> u8 {
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("file://")
        || lower.contains("docs.rs")
        || lower.contains("rust-lang.org")
        || lower.contains("python.org")
        || lower.contains("nodejs.org")
        || lower.contains("mozilla.org")
        || lower.contains("w3.org")
        || lower.contains("ietf.org")
        || lower.contains("rfc-editor.org")
        || lower.contains("developer.")
        || lower.contains("/docs")
        || lower.contains("docs.")
    {
        0
    } else if lower.contains("github.com")
        || lower.contains("github.io")
        || lower.contains("gitlab.")
        || lower.contains("npmjs.com/package")
        || lower.contains("crates.io/crates")
    {
        1
    } else {
        2
    }
}

fn source_badge(url: &str) -> &'static str {
    match official_source_rank(url) {
        0 => " [official]",
        1 => " [primary]",
        _ => "",
    }
}

fn url_matches_site_filters(url: &str, site_filters: &[String]) -> bool {
    if site_filters.is_empty() {
        return true;
    }
    let lower_url = url.to_ascii_lowercase();
    site_filters.iter().any(|filter| {
        let filter = normalize_site_filter(filter);
        !filter.is_empty() && lower_url.contains(&filter)
    })
}

fn merged_site_filters(query: &str, explicit_filters: &[String]) -> Vec<String> {
    let mut filters = explicit_filters.to_vec();
    filters.extend(extract_inline_site_filters(query));
    filters.sort_by_key(|filter| normalize_site_filter(filter));
    filters.dedup_by(|a, b| normalize_site_filter(a) == normalize_site_filter(b));
    filters
}

fn extract_inline_site_filters(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter_map(|part| {
            let lower = part.to_ascii_lowercase();
            let raw = lower.strip_prefix("site:")?;
            let cleaned = raw
                .trim_matches(|c: char| {
                    c == '"' || c == '\'' || c == '(' || c == ')' || c == '[' || c == ']'
                })
                .trim_matches('/');
            (!cleaned.is_empty()).then_some(cleaned.to_string())
        })
        .collect()
}

fn normalize_site_filter(filter: &str) -> String {
    let lower = filter.to_ascii_lowercase();
    lower
        .trim()
        .trim_start_matches("site:")
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.")
        .trim_matches('/')
        .to_ascii_lowercase()
}

fn collect_code_examples(docs: &[ResearchDocument]) -> Vec<String> {
    let mut examples = Vec::new();
    for doc in docs {
        for block in extract_code_like_lines(&doc.content).into_iter().take(3) {
            examples.push(format!("From {}:\n```text\n{}\n```", doc.url, block));
            if examples.len() >= 8 {
                return examples;
            }
        }
    }
    examples
}

fn conflict_and_staleness_notes(docs: &[ResearchDocument]) -> String {
    let mut notes = Vec::new();
    if docs.iter().any(|doc| doc.quality_score < 0.35) {
        notes.push("- One or more sources extracted sparse content; use rendered follow-up before relying on hidden examples.");
    }
    if docs.iter().any(|doc| doc.from_cache) {
        notes.push("- Some sources came from the local research cache; retrieval timestamps are listed in the source list.");
    }
    if docs.iter().any(|doc| official_source_rank(&doc.url) > 0) {
        notes.push("- At least one source is not obviously official/primary; prefer primary docs when details conflict.");
    }
    if notes.is_empty() {
        notes.push("- No obvious source conflicts detected by the native extractor.");
    }
    notes.join("\n") + "\n"
}

fn implementation_implications(docs: &[ResearchDocument]) -> String {
    let mut lines = vec![
        "- Prefer APIs and options directly documented by the primary sources above.".to_string(),
        "- When editing code, cite the source URL in the reasoning trail and prefer source behavior over prior model knowledge if they differ.".to_string(),
    ];
    if docs.iter().any(|doc| doc.rendered_fallback_used) {
        lines.push(
            "- Rendered fallback found additional content; preserve browser-backed follow-up for this area."
                .to_string(),
        );
    }
    if docs.iter().any(|doc| doc.quality_score < 0.35) {
        lines.push(
            "- Do not implement from sparse extraction alone; fetch a rendered page or local package docs first."
                .to_string(),
        );
    }
    lines.join("\n") + "\n"
}

fn extract_code_like_lines(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        let looks_code = trimmed.starts_with("use ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("function ")
            || trimmed.contains("::")
            || trimmed.contains("=>");
        if looks_code {
            current.push(trimmed.to_string());
            if current.len() >= 12 {
                blocks.push(current.join("\n"));
                current.clear();
            }
        } else if !current.is_empty() {
            blocks.push(current.join("\n"));
            current.clear();
        }
    }
    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }
    blocks
}

fn excerpt(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!(
            "{}...",
            mangocode_core::truncate::truncate_bytes_prefix(text, max)
        )
    }
}

fn verification_status(verified_at_read_time: bool) -> &'static str {
    if verified_at_read_time {
        "reachable at read time"
    } else {
        "not revalidated"
    }
}

fn research_cache_dir() -> PathBuf {
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".mangocode");
    dir.push("research");
    dir.push(RESEARCH_CACHE_VERSION);
    dir
}

fn research_cache_path(url: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    research_cache_dir().join(format!("{}.json", hex::encode(hasher.finalize())))
}

fn load_cached_document(url: &str) -> Option<ResearchDocument> {
    let raw = std::fs::read_to_string(research_cache_path(url)).ok()?;
    let mut doc: ResearchDocument = serde_json::from_str(&raw).ok()?;
    if doc.url != url || cache_is_stale(&doc.retrieved_at) {
        return None;
    }
    doc.from_cache = true;
    Some(doc)
}

fn save_cached_document(doc: &ResearchDocument) -> std::io::Result<()> {
    std::fs::create_dir_all(research_cache_dir())?;
    std::fs::write(
        research_cache_path(&doc.url),
        serde_json::to_string_pretty(doc).unwrap_or_default(),
    )
}

fn research_index_path() -> PathBuf {
    research_cache_dir().join("source-index.json")
}

fn load_research_index() -> Vec<ResearchIndexEntry> {
    std::fs::read_to_string(research_index_path())
        .ok()
        .and_then(|raw| serde_json::from_str::<Vec<ResearchIndexEntry>>(&raw).ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| !cache_is_stale(&entry.retrieved_at))
        .collect()
}

fn save_research_index(entries: &[ResearchIndexEntry]) -> std::io::Result<()> {
    std::fs::create_dir_all(research_cache_dir())?;
    std::fs::write(
        research_index_path(),
        serde_json::to_string_pretty(entries).unwrap_or_default(),
    )
}

fn upsert_research_index(doc: &ResearchDocument) -> std::io::Result<()> {
    if doc.quality_score < 0.12 || doc.content.trim().is_empty() {
        return Ok(());
    }
    let mut entries = load_research_index();
    entries.retain(|entry| entry.url != doc.url);
    entries.push(ResearchIndexEntry {
        url: doc.url.clone(),
        title: doc.title.clone(),
        content_excerpt: excerpt(&doc.content, 12_000),
        content_type: doc.content_type.clone(),
        retrieved_at: doc.retrieved_at.clone(),
        quality_score: doc.quality_score,
        rendered_fallback_used: doc.rendered_fallback_used,
        terms: query_terms(&format!(
            "{} {} {}",
            doc.title.as_deref().unwrap_or(""),
            doc.url,
            doc.content
        ))
        .into_iter()
        .take(300)
        .collect(),
    });
    entries.sort_by(|a, b| {
        b.quality_score
            .total_cmp(&a.quality_score)
            .then_with(|| b.retrieved_at.cmp(&a.retrieved_at))
    });
    entries.truncate(2_000);
    save_research_index(&entries)
}

fn search_research_index(query: &str, max_results: usize) -> Vec<ResearchIndexEntry> {
    let terms = query_terms(query);
    if terms.is_empty() {
        return Vec::new();
    }
    let mut scored = load_research_index()
        .into_iter()
        .filter_map(|entry| {
            let score = score_index_entry(&entry, &terms);
            (score > 0.0).then_some((score, entry))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| official_source_rank(&a.1.url).cmp(&official_source_rank(&b.1.url)))
            .then_with(|| b.1.quality_score.total_cmp(&a.1.quality_score))
    });
    scored
        .into_iter()
        .take(max_results)
        .map(|(_, entry)| entry)
        .collect()
}

fn score_index_entry(entry: &ResearchIndexEntry, terms: &[String]) -> f32 {
    let title = entry.title.as_deref().unwrap_or("").to_ascii_lowercase();
    let url = entry.url.to_ascii_lowercase();
    let excerpt = entry.content_excerpt.to_ascii_lowercase();
    let indexed_terms = entry.terms.iter().collect::<std::collections::HashSet<_>>();
    let mut score = 0.0;
    for term in terms {
        if title.contains(term) {
            score += 5.0;
        }
        if url.contains(term) {
            score += 3.0;
        }
        if indexed_terms.contains(term) {
            score += 1.5;
        } else if excerpt.contains(term) {
            score += 1.0;
        }
    }
    score
        + entry.quality_score
        + if official_source_rank(&entry.url) == 0 {
            1.0
        } else {
            0.0
        }
}

fn cache_is_stale(retrieved_at: &str) -> bool {
    let Ok(dt) = chrono::DateTime::parse_from_rfc3339(retrieved_at) else {
        return true;
    };
    let age = chrono::Utc::now().signed_duration_since(dt.with_timezone(&chrono::Utc));
    age.num_seconds() < 0 || age.num_seconds() as u64 > RESEARCH_CACHE_TTL_SECS
}

fn encode_query(query: &str) -> String {
    query
        .chars()
        .map(|ch| match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => ch.to_string(),
            ' ' => "+".to_string(),
            _ => ch
                .to_string()
                .as_bytes()
                .iter()
                .map(|b| format!("%{:02X}", b))
                .collect::<Vec<_>>()
                .join(""),
        })
        .collect::<Vec<_>>()
        .join("")
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
        .filter(|term| term.len() >= 3 && !is_search_stopword(term))
        .map(str::to_string)
        .collect()
}

fn is_search_stopword(term: &str) -> bool {
    matches!(
        term,
        "docs"
            | "documentation"
            | "api"
            | "guide"
            | "tutorial"
            | "example"
            | "examples"
            | "about"
            | "overview"
            | "info"
            | "information"
            | "how"
            | "use"
            | "using"
            | "with"
            | "for"
            | "the"
            | "and"
    )
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(decoded) = u8::from_str_radix(hex, 16) {
                    out.push(decoded);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn apply_site_filters(query: &str, filters: &[String]) -> String {
    if filters.is_empty() {
        return query.to_string();
    }
    let sites = filters
        .iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| format!("site:{}", s.trim()))
        .collect::<Vec<_>>()
        .join(" OR ");
    format!("{} {}", query, sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_scores_sparse_pages_low() {
        assert!(score_content_quality("loading") < 0.35);
    }

    #[test]
    fn official_sources_rank_first() {
        assert_eq!(official_source_rank("https://docs.rs/foo"), 0);
        assert_eq!(official_source_rank("file://C:/project/docs/api.md"), 0);
        assert_eq!(official_source_rank("https://github.com/tokio-rs/tokio"), 1);
        assert_eq!(official_source_rank("https://example-blog.com/foo"), 2);
    }

    #[test]
    fn html_extraction_keeps_text() {
        let text = html_to_markdownish(
            "<html><title>T</title><body><h1>Hello</h1><p>A &amp; B</p></body></html>",
        );
        assert!(text.contains("Hello"));
        assert!(text.contains("A & B"));
    }

    #[test]
    fn discovers_matching_local_docs() {
        let tmp = tempfile::tempdir().expect("tmp");
        let docs = tmp.path().join("docs");
        std::fs::create_dir_all(&docs).expect("docs");
        std::fs::write(
            docs.join("routing.md"),
            "# Routing\n\nAttachmentRouter handles images and PDFs.",
        )
        .expect("write");

        let hits = discover_local_docs(tmp.path(), "AttachmentRouter PDF routing", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].contains("routing.md"));
    }

    #[test]
    fn skips_manifest_noise_for_generic_doc_queries() {
        let tmp = tempfile::tempdir().expect("tmp");
        std::fs::write(
            tmp.path().join("Cargo.toml"),
            "[dependencies]\ntokio-util = \"0.7\"",
        )
        .expect("write manifest");
        std::fs::write(
            tmp.path().join("guide.md"),
            "# tokio_util CancellationToken documentation\n\nUse tokio_util::sync::CancellationToken for cooperative cancellation.",
        )
        .expect("write guide");

        let hits = discover_local_docs(tmp.path(), "tokio_util CancellationToken documentation", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].contains("guide.md"));
    }

    #[test]
    fn extracts_duckduckgo_html_result_urls() {
        let html = r#"<a class="result__a" href="/l/?kh=-1&uddg=https%3A%2F%2Fdocs.rs%2Fserde">serde docs</a>"#;
        let urls = extract_search_result_urls(html, 5);
        assert_eq!(urls, vec!["https://docs.rs/serde"]);
    }

    #[test]
    fn curated_candidates_include_primary_sources() {
        let urls = curated_official_candidates("rust serde docs");
        assert!(urls.iter().any(|url| url.contains("docs.rs/serde")));
    }

    #[test]
    fn curated_tokio_candidates_include_cancellation_docs() {
        let urls = curated_official_candidates("tokio task cancellation docs.rs");
        assert!(urls
            .iter()
            .any(|url| url.contains("struct.JoinHandle.html")));
        assert!(urls
            .iter()
            .any(|url| url.contains("struct.AbortHandle.html")));
        assert!(urls
            .iter()
            .any(|url| url.contains("CancellationToken.html")));
    }

    #[test]
    fn site_filters_remove_non_matching_docsearch_results() {
        assert!(url_matches_site_filters(
            "https://docs.rs/tokio/latest/tokio/",
            &["docs.rs".to_string()]
        ));
        assert!(!url_matches_site_filters(
            "https://github.com/tokio-rs/tokio",
            &["docs.rs".to_string()]
        ));
    }

    #[test]
    fn inline_site_filters_disable_local_docs_and_filter_results() {
        let filters = merged_site_filters("tokio CancellationToken site:docs.rs", &[]);
        assert_eq!(filters, vec!["docs.rs".to_string()]);
        assert!(url_matches_site_filters(
            "https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html",
            &filters
        ));
        assert!(!url_matches_site_filters(
            "file://C:/project/docs/tokio.md",
            &filters
        ));
    }

    #[test]
    fn docs_rs_fallback_suggests_valid_tokio_cancellation_pages() {
        let doc = docs_rs_fallback_document(
            "https://docs.rs/tokio/latest/tokio/task/struct.Handle.html",
            "HTTP 404",
        );
        assert!(doc.content.contains("struct.JoinHandle.html"));
        assert!(doc.content.contains("struct.AbortHandle.html"));
        assert!(doc.content.contains("CancellationToken.html"));
        assert!(!doc.verified_at_read_time);
    }

    #[test]
    fn source_index_scores_relevant_official_docs() {
        let entry = ResearchIndexEntry {
            url: "https://docs.rs/serde/latest/serde/".to_string(),
            title: Some("serde documentation".to_string()),
            content_excerpt: "Serialize and Deserialize derive macros for Rust data structures."
                .to_string(),
            content_type: "text/html".to_string(),
            retrieved_at: chrono::Utc::now().to_rfc3339(),
            quality_score: 0.9,
            rendered_fallback_used: false,
            terms: query_terms("serde serialize deserialize rust"),
        };
        let score = score_index_entry(&entry, &query_terms("serde deserialize"));
        assert!(score > 5.0);
    }

    #[test]
    fn rendered_fetch_format_adds_metadata_and_heading() {
        let rendered = format_rendered_fetch(
            "https://example.com/docs",
            "hello from browser",
            "metadata",
            "markdown",
        );
        assert!(rendered.contains("Source: https://example.com/docs"));
        assert!(rendered.contains("Extraction: rendered-browser"));
        assert!(rendered.contains("# RenderedFetch: https://example.com/docs"));
        assert!(rendered.contains("hello from browser"));
    }

    #[test]
    fn sparse_cached_doc_needs_rendered_upgrade() {
        let doc = ResearchDocument {
            url: "https://example.com".to_string(),
            title: None,
            content: "loading".to_string(),
            content_type: "text/html".to_string(),
            retrieved_at: chrono::Utc::now().to_rfc3339(),
            quality_score: 0.1,
            rendered_fallback_used: false,
            verified_at_read_time: true,
            from_cache: true,
        };
        assert!(should_upgrade_cached_doc(&doc));
    }
}
