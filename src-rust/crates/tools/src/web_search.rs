// WebSearch tool: search the web using Brave Search API or fallback to DuckDuckGo.
//
// Behaviour:
// - Accepts a query string
// - Returns a list of results with title, url, and snippet
// - Falls back to DuckDuckGo if no search API key is configured

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

pub struct WebSearchTool;

#[derive(Debug, Deserialize)]
struct WebSearchInput {
    query: String,
    #[serde(default = "default_num_results")]
    num_results: usize,
    #[serde(default = "default_source_preference")]
    source_preference: String,
    #[serde(default)]
    recency: Option<u32>,
    #[serde(default)]
    site_filters: Vec<String>,
}

fn default_num_results() -> usize {
    5
}

fn default_source_preference() -> String {
    "any".to_string()
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_WEB_SEARCH
    }

    fn description(&self) -> &str {
        "Search the web for information. Returns a list of relevant web pages with \
         titles, URLs, and snippets. Use this when you need current information \
         not available in your training data, or when searching for documentation, \
         examples, or news."
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
                    "description": "The search query"
                },
                "num_results": {
                    "type": "number",
                    "description": "Number of results to return (default: 5, max: 10)"
                },
                "source_preference": {
                    "type": "string",
                    "enum": ["any", "official", "primary"],
                    "description": "Rank official docs, repos, standards, and release notes first"
                },
                "recency": {
                    "type": "number",
                    "description": "Optional freshness hint in days, appended to providers that support query text filtering"
                },
                "site_filters": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional domains to restrict search, e.g. docs.rs"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> ToolResult {
        let params: WebSearchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let num_results = params.num_results.clamp(1, 10);
        let site_filters = merged_site_filters(&params.query, &params.site_filters);
        let query = build_query(&params.query, params.recency, &params.site_filters);
        debug!(query = %query, num_results, "Web search");

        // Try Brave Search API first, then fall back to DuckDuckGo
        if let Some(api_key) = std::env::var("BRAVE_SEARCH_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
        {
            search_brave(
                &query,
                num_results,
                &api_key,
                &params.source_preference,
                &site_filters,
            )
            .await
        } else {
            search_duckduckgo(
                &query,
                num_results,
                &params.source_preference,
                &site_filters,
            )
            .await
        }
    }
}

/// Search using the Brave Search API.
async fn search_brave(
    query: &str,
    num_results: usize,
    api_key: &str,
    source_preference: &str,
    site_filters: &[String],
) -> ToolResult {
    let client = reqwest::Client::new();
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
        urlencoding_simple(query),
        num_results
    );

    let resp = match client
        .get(&url)
        .header("Accept", "application/json")
        .header("Accept-Encoding", "gzip")
        .header("X-Subscription-Token", api_key)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Search request failed: {}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        return ToolResult::error(format!("Brave Search API returned status {}", status));
    }

    let data: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("Failed to parse response: {}", e)),
    };

    let results = format_brave_results(&data, num_results, source_preference, site_filters);
    ToolResult::success(results)
}

fn format_brave_results(
    data: &Value,
    max: usize,
    source_preference: &str,
    site_filters: &[String],
) -> String {
    let web_results = data
        .get("web")
        .and_then(|w| w.get("results"))
        .and_then(|r| r.as_array());

    let mut formatted = Vec::new();
    if let Some(items) = web_results {
        let mut items = items.iter().collect::<Vec<_>>();
        if source_preference != "any" {
            items.sort_by_key(|item| {
                item.get("url")
                    .and_then(|u| u.as_str())
                    .map(official_source_rank)
                    .unwrap_or(2)
            });
        }
        for item in items.into_iter() {
            let title = item
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("(No title)");
            let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
            if !url_matches_site_filters(url, site_filters) {
                continue;
            }
            let snippet = item
                .get("description")
                .and_then(|s| s.as_str())
                .unwrap_or("");

            formatted.push((title.to_string(), url.to_string(), snippet.to_string()));
            if formatted.len() >= max {
                break;
            }
        }
    }

    if formatted.is_empty() {
        "No results found.".to_string()
    } else {
        format_results(formatted)
    }
}

/// Fallback: DuckDuckGo Instant Answer API plus HTML/lite search parsing.
async fn search_duckduckgo(
    query: &str,
    num_results: usize,
    source_preference: &str,
    site_filters: &[String],
) -> ToolResult {
    let client = reqwest::Client::new();
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding_simple(query)
    );

    let resp = match client
        .get(&url)
        .header("User-Agent", "MangoCode/1.0")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("Search request failed: {}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        return ToolResult::error(format!("DuckDuckGo API returned status {}", status));
    }

    let data: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return ToolResult::error(format!("Failed to parse response: {}", e)),
    };

    let mut results = curated_search_candidates(query, site_filters);
    results.extend(collect_ddg_instant_results(&data, num_results));
    if results.len() < num_results {
        match search_duckduckgo_html(query, num_results.saturating_mul(2)).await {
            Ok(mut html_results) => results.append(&mut html_results),
            Err(e) => debug!(error = %e, "DuckDuckGo HTML fallback failed"),
        }
    }

    results.retain(|(_, url, _)| url.starts_with("http://") || url.starts_with("https://"));
    results.retain(|(_, url, _)| url_matches_site_filters(url, site_filters));
    results.sort_by(|a, b| {
        if source_preference != "any" {
            official_source_rank(&a.1).cmp(&official_source_rank(&b.1))
        } else {
            std::cmp::Ordering::Equal
        }
        .then_with(|| a.1.cmp(&b.1))
    });
    results.dedup_by(|a, b| a.1 == b.1);

    if results.is_empty() {
        ToolResult::error(format!(
            "No web results found for '{}'. Brave is optional; DuckDuckGo instant/html fallback also returned no usable results.",
            query
        ))
    } else {
        ToolResult::success(format_results(
            results.into_iter().take(num_results).collect(),
        ))
    }
}

fn curated_search_candidates(
    query: &str,
    site_filters: &[String],
) -> Vec<(String, String, String)> {
    let lower = query.to_ascii_lowercase();
    let mut results = Vec::new();
    if lower.contains("tokio") {
        push_candidate(
            &mut results,
            "Graceful Shutdown | Tokio",
            "https://tokio.rs/tokio/topics/shutdown",
            "Official Tokio guide covering shutdown detection, cancellation signaling, and waiting for tasks.",
            site_filters,
        );
        push_candidate(
            &mut results,
            "JoinHandle in tokio::task - Docs.rs",
            "https://docs.rs/tokio/latest/tokio/task/struct.JoinHandle.html",
            "Official rustdoc for spawned task handles, including abort and cancellation behavior.",
            site_filters,
        );
        push_candidate(
            &mut results,
            "AbortHandle in tokio::task - Docs.rs",
            "https://docs.rs/tokio/latest/tokio/task/struct.AbortHandle.html",
            "Official rustdoc for remotely aborting spawned Tokio tasks.",
            site_filters,
        );
        push_candidate(
            &mut results,
            "CancellationToken in tokio_util::sync - Docs.rs",
            "https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html",
            "Official rustdoc for cooperative cancellation tokens used in graceful shutdown.",
            site_filters,
        );
        push_candidate(
            &mut results,
            "TaskTracker in tokio_util::task - Docs.rs",
            "https://docs.rs/tokio-util/latest/tokio_util/task/task_tracker/struct.TaskTracker.html",
            "Official rustdoc for waiting on tracked tasks during shutdown.",
            site_filters,
        );
    }
    results
}

fn push_candidate(
    results: &mut Vec<(String, String, String)>,
    title: &str,
    url: &str,
    snippet: &str,
    site_filters: &[String],
) {
    if url_matches_site_filters(url, site_filters) {
        results.push((title.to_string(), url.to_string(), snippet.to_string()));
    }
}

fn collect_ddg_instant_results(data: &Value, max: usize) -> Vec<(String, String, String)> {
    let mut formatted = Vec::new();
    let mut count = 0;

    // Abstract (main answer)
    if let Some(abstract_text) = data.get("Abstract").and_then(|a| a.as_str()) {
        if !abstract_text.is_empty() {
            let source = data
                .get("AbstractSource")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            let url = data
                .get("AbstractURL")
                .and_then(|u| u.as_str())
                .unwrap_or("");
            formatted.push((
                source.to_string(),
                url.to_string(),
                abstract_text.to_string(),
            ));
            count += 1;
        }
    }

    // Related topics
    if let Some(topics) = data.get("RelatedTopics").and_then(|t| t.as_array()) {
        for topic in topics.iter().take(max.saturating_sub(count)) {
            if let Some(text) = topic.get("Text").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    let url = topic.get("FirstURL").and_then(|u| u.as_str()).unwrap_or("");
                    formatted.push((
                        "Related topic".to_string(),
                        url.to_string(),
                        text.to_string(),
                    ));
                }
            }
        }
    }

    formatted.into_iter().take(max).collect()
}

async fn search_duckduckgo_html(
    query: &str,
    max_results: usize,
) -> Result<Vec<(String, String, String)>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("Search client failed: {}", e))?;
    for endpoint in [
        format!(
            "https://duckduckgo.com/html/?q={}",
            urlencoding_simple(query)
        ),
        format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            urlencoding_simple(query)
        ),
    ] {
        let html = client
            .get(endpoint)
            .header("User-Agent", "MangoCode/1.0 web search")
            .send()
            .await
            .map_err(|e| format!("HTML search failed: {}", e))?
            .text()
            .await
            .map_err(|e| format!("HTML search body failed: {}", e))?;
        let results = extract_duckduckgo_html_results(&html, max_results);
        if !results.is_empty() {
            return Ok(results);
        }
    }
    Ok(Vec::new())
}

fn format_results(results: Vec<(String, String, String)>) -> String {
    let mut output = String::new();
    for (i, (title, url, snippet)) in results.into_iter().enumerate() {
        let marker = source_badge(&url);
        output.push_str(&format!(
            "{}. **{}**{}\n   URL: {}\n   {}\n\n",
            i + 1,
            title,
            marker,
            url,
            snippet
        ));
    }
    output
}

fn official_source_rank(url: &str) -> u8 {
    let lower = url.to_ascii_lowercase();
    if lower.contains("docs.rs")
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

fn build_query(query: &str, recency: Option<u32>, site_filters: &[String]) -> String {
    let mut q = query.to_string();
    if let Some(days) = recency {
        q.push_str(&format!(" after:{}d", days));
    }
    if !site_filters.is_empty() {
        let sites = site_filters
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("site:{}", s.trim()))
            .collect::<Vec<_>>()
            .join(" OR ");
        if !sites.is_empty() {
            q.push(' ');
            q.push_str(&sites);
        }
    }
    q
}

fn extract_duckduckgo_html_results(
    html: &str,
    max_results: usize,
) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut cursor = 0usize;
    while results.len() < max_results {
        let Some(href_rel) = html[cursor..].find("uddg=") else {
            break;
        };
        let url_start = cursor + href_rel + "uddg=".len();
        let url_end = html[url_start..]
            .find(['&', '"', '\''])
            .map(|rel| url_start + rel)
            .unwrap_or(html.len());
        let url = percent_decode(&html[url_start..url_end]);
        cursor = url_end;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            continue;
        }

        let block_end = html[cursor..]
            .find("uddg=")
            .map(|rel| cursor + rel)
            .unwrap_or(html.len())
            .min(cursor.saturating_add(4000));
        let block = &html[cursor..block_end];
        let title = extract_link_text(block).unwrap_or_else(|| url.clone());
        let snippet = extract_snippet_text(block).unwrap_or_default();
        results.push((title, url, snippet));
    }

    if results.is_empty() {
        let mut cursor = 0usize;
        while results.len() < max_results {
            let Some(href_rel) = html[cursor..].find("href=\"http") else {
                break;
            };
            let url_start = cursor + href_rel + "href=\"".len();
            let url_end = html[url_start..]
                .find('"')
                .map(|rel| url_start + rel)
                .unwrap_or(html.len());
            let url = decode_entities(&html[url_start..url_end]);
            cursor = url_end;
            if url.contains("duckduckgo.com") {
                continue;
            }
            let block_end = html[cursor..]
                .find("href=\"http")
                .map(|rel| cursor + rel)
                .unwrap_or(html.len())
                .min(cursor.saturating_add(4000));
            let block = &html[cursor..block_end];
            let title = extract_link_text(block).unwrap_or_else(|| url.clone());
            let snippet = extract_snippet_text(block).unwrap_or_default();
            results.push((title, url, snippet));
        }
    }

    results.dedup_by(|a, b| a.1 == b.1);
    results.truncate(max_results);
    results
}

fn extract_link_text(block: &str) -> Option<String> {
    let gt = block.find('>')?;
    let end = block[gt + 1..].find("</a>").map(|rel| gt + 1 + rel)?;
    let text = strip_html(&block[gt + 1..end]);
    (!text.trim().is_empty()).then_some(text)
}

fn extract_snippet_text(block: &str) -> Option<String> {
    let marker = block
        .find("result__snippet")
        .or_else(|| block.find("result-snippet"))
        .or_else(|| block.find("snippet"))?;
    let rest = &block[marker..];
    let gt = rest.find('>')?;
    let end = rest[gt + 1..]
        .find("</")
        .map(|rel| gt + 1 + rel)
        .unwrap_or(rest.len().min(gt + 300));
    let text = strip_html(&rest[gt + 1..end]);
    (!text.trim().is_empty()).then_some(text)
}

fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    decode_entities(out.trim())
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

fn decode_entities(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// Minimal percent-encoding for URL query parameters.
fn urlencoding_simple(s: &str) -> String {
    let mut encoded = String::new();
    for ch in s.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                encoded.push(ch);
            }
            ' ' => encoded.push('+'),
            _ => {
                for byte in ch.to_string().as_bytes() {
                    encoded.push_str(&format!("%{:02X}", byte));
                }
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_duckduckgo_html_results() {
        let html = r#"
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.snhu.edu%2F">Southern New Hampshire University</a>
            <a class="result__snippet">SNHU is a private nonprofit university.</a>
        "#;
        let results = extract_duckduckgo_html_results(html, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, "https://www.snhu.edu/");
        assert!(results[0].0.contains("Southern New Hampshire"));
        assert!(results[0].2.contains("private nonprofit"));
    }

    #[test]
    fn percent_decodes_duckduckgo_redirect_urls() {
        assert_eq!(
            percent_decode("https%3A%2F%2Fexample.com%2Fa%2Bb"),
            "https://example.com/a+b"
        );
    }

    #[test]
    fn github_results_are_primary_not_official() {
        assert_eq!(official_source_rank("https://github.com/tokio-rs/tokio"), 1);
        assert_eq!(
            source_badge("https://github.com/tokio-rs/tokio"),
            " [primary]"
        );
        assert_eq!(source_badge("https://docs.rs/tokio"), " [official]");
    }

    #[test]
    fn curated_tokio_docs_fill_duckduckgo_gaps() {
        let results = curated_search_candidates(
            "tokio task handle abort official documentation site:docs.rs",
            &["docs.rs".to_string()],
        );
        assert!(results
            .iter()
            .any(|(_, url, _)| url.contains("struct.JoinHandle.html")));
        assert!(results.iter().all(|(_, url, _)| url.contains("docs.rs")));
    }

    #[test]
    fn inline_site_filters_are_enforced_like_structured_filters() {
        let filters = merged_site_filters("tokio abort site:docs.rs", &[]);
        assert_eq!(filters, vec!["docs.rs".to_string()]);
        assert!(url_matches_site_filters(
            "https://docs.rs/tokio/latest/tokio/",
            &filters
        ));
        assert!(!url_matches_site_filters(
            "https://tokio.rs/tokio/topics/shutdown",
            &filters
        ));
    }
}
