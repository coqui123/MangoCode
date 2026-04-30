use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

#[cfg_attr(not(feature = "browser"), allow(dead_code))]
#[derive(Debug, Clone, Deserialize)]
struct BrowserInput {
    action: String,
    url: Option<String>,
    selector: Option<String>,
    text: Option<String>,
    script: Option<String>,
}

pub struct BrowserTool;

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "Browser"
    }

    fn description(&self) -> &str {
        "Control a persistent headless browser session. Navigate to URLs, take screenshots, extract text/Markdown, click elements, fill forms, evaluate JavaScript, apply extraction recipes, or close the session."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Dangerous
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "enum": ["navigate", "screenshot", "extract_text", "extract_markdown", "click", "type", "evaluate", "expand", "close"] },
                "url": { "type": "string" },
                "selector": { "type": "string", "description": "CSS selector for click/type" },
                "text": { "type": "string", "description": "Text to type" },
                "script": { "type": "string", "description": "JavaScript to evaluate" }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: BrowserInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid Browser input: {}", e)),
        };

        let desc = format!("browser: {}", params.action);
        if let Err(e) = ctx.check_permission(self.name(), &desc, false) {
            return ToolResult::error(e.to_string());
        }

        execute_browser_action(params, ctx).await
    }
}

#[cfg(not(feature = "browser"))]
async fn execute_browser_action(_params: BrowserInput, _ctx: &ToolContext) -> ToolResult {
    ToolResult::error(
        "The browser feature is not enabled in this build. Recompile with --features mangocode-tools/browser to enable it.",
    )
}

#[cfg(feature = "browser")]
static BROWSER_SESSIONS: once_cell::sync::Lazy<
    dashmap::DashMap<String, std::sync::Arc<tokio::sync::Mutex<BrowserSession>>>,
> = once_cell::sync::Lazy::new(dashmap::DashMap::new);

#[cfg(feature = "browser")]
async fn execute_browser_action(params: BrowserInput, ctx: &ToolContext) -> ToolResult {
    if params.action == "close" {
        return match close_browser_session(&ctx.session_id).await {
            Ok(closed) if closed => ToolResult::success("Closed persistent browser session."),
            Ok(_) => ToolResult::success("No persistent browser session was open."),
            Err(e) => ToolResult::error(e),
        };
    }

    let session = match browser_session(&ctx.session_id).await {
        Ok(session) => session,
        Err(e) => return ToolResult::error(e),
    };

    let action_result = session.lock().await.run_action(&params).await;
    match action_result {
        Ok(v) => ToolResult::success(v),
        Err(e) if is_recoverable_browser_error(&e) => {
            let _ = reset_browser_session(&ctx.session_id).await;
            match browser_session(&ctx.session_id).await {
                Ok(session) => match session.lock().await.run_action(&params).await {
                    Ok(v) => ToolResult::success(v),
                    Err(retry_error) => ToolResult::error(format!(
                        "{} (browser session was reset and retried once, but failed again: {})",
                        e, retry_error
                    )),
                },
                Err(reset_error) => ToolResult::error(format!(
                    "{} (browser session reset failed: {})",
                    e, reset_error
                )),
            }
        }
        Err(e) => ToolResult::error(e),
    }
}

#[cfg(feature = "browser")]
pub async fn rendered_extract_for_research(url: &str) -> Result<String, String> {
    match rendered_extract_for_research_once("__mangocode_research__", url).await {
        Ok(markdown) => Ok(markdown),
        Err(error) if is_recoverable_browser_error(&error) => {
            let _ = reset_browser_session("__mangocode_research__").await;
            rendered_extract_for_research_once("__mangocode_research__", url)
                .await
                .map_err(|retry_error| {
                    format!(
                        "{} (browser session was reset and retried once, but failed again: {})",
                        error, retry_error
                    )
                })
        }
        Err(error) => Err(error),
    }
}

#[cfg(not(feature = "browser"))]
pub async fn rendered_extract_for_research(_url: &str) -> Result<String, String> {
    Err("browser feature is not enabled".to_string())
}

#[cfg(feature = "browser")]
async fn browser_session(
    session_id: &str,
) -> Result<std::sync::Arc<tokio::sync::Mutex<BrowserSession>>, String> {
    if let Some(session) = BROWSER_SESSIONS.get(session_id) {
        return Ok(session.value().clone());
    }

    use chromiumoxide::browser::{Browser, BrowserConfig};
    use futures::StreamExt;

    let config = BrowserConfig::builder()
        .build()
        .map_err(|e| format!("Failed to build browser config: {}", e))?;
    let (browser, mut handler) = Browser::launch(config).await.map_err(|e| {
        format!(
            "Failed to launch Chromium. Ensure Chrome/Chromium is installed and discoverable: {}",
            e
        )
    })?;

    let handler_task = tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if event.is_err() {
                break;
            }
        }
    });

    let session = std::sync::Arc::new(tokio::sync::Mutex::new(BrowserSession {
        browser,
        page: None,
        current_url: None,
        handler_task,
    }));
    BROWSER_SESSIONS.insert(session_id.to_string(), session.clone());
    Ok(session)
}

#[cfg(feature = "browser")]
async fn close_browser_session(session_id: &str) -> Result<bool, String> {
    let Some((_, session)) = BROWSER_SESSIONS.remove(session_id) else {
        return Ok(false);
    };
    session.lock().await.close().await;
    Ok(true)
}

#[cfg(feature = "browser")]
async fn reset_browser_session(session_id: &str) -> Result<(), String> {
    let _ = close_browser_session(session_id).await?;
    Ok(())
}

#[cfg(feature = "browser")]
async fn rendered_extract_for_research_once(session_id: &str, url: &str) -> Result<String, String> {
    let session = browser_session(session_id).await?;
    let mut guard = session.lock().await;
    guard.extract_research_markdown(url).await
}

#[cfg(feature = "browser")]
fn is_recoverable_browser_error(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    [
        "oneshot canceled",
        "receiver is gone",
        "send failed",
        "target closed",
        "session closed",
        "channel closed",
        "browser has disconnected",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

#[cfg(feature = "browser")]
struct BrowserSession {
    browser: chromiumoxide::browser::Browser,
    page: Option<chromiumoxide::Page>,
    current_url: Option<String>,
    handler_task: tokio::task::JoinHandle<()>,
}

#[cfg(feature = "browser")]
impl BrowserSession {
    async fn close(&mut self) {
        self.page = None;
        let _ = self.browser.close().await;
        let _ = self.browser.wait().await;
        self.handler_task.abort();
    }

    async fn run_action(&mut self, input: &BrowserInput) -> Result<String, String> {
        match input.action.as_str() {
            "navigate" => {
                let url = input
                    .url
                    .as_deref()
                    .ok_or_else(|| "navigate requires 'url'".to_string())?;
                let page = self.navigate(url).await?;
                let title = page
                    .get_title()
                    .await
                    .map_err(|e| format!("Failed to read page title: {}", e))?
                    .unwrap_or_else(|| "(untitled)".to_string());
                Ok(format!(
                    "Navigated persistent browser session to {} (title: {})",
                    self.current_url.as_deref().unwrap_or(url),
                    title
                ))
            }
            "screenshot" => self.screenshot(input.url.as_deref()).await,
            "extract_text" => {
                let page = self.page_for(input.url.as_deref()).await?;
                self.apply_recipes(&page).await?;
                let text = self.extract_inner_text(&page).await?;
                self.refresh_current_url(&page).await;
                Ok(text)
            }
            "extract_markdown" => {
                let page = self.page_for(input.url.as_deref()).await?;
                let markdown = self.extract_adaptive_markdown(&page).await?;
                self.refresh_current_url(&page).await;
                Ok(markdown)
            }
            "expand" => {
                let page = self.page_for(input.url.as_deref()).await?;
                let summary = self.apply_recipes(&page).await?;
                self.refresh_current_url(&page).await;
                Ok(summary)
            }
            "click" => self.click(input).await,
            "type" => self.type_text(input).await,
            "evaluate" => self.evaluate(input).await,
            other => Err(format!(
                "Unknown action '{}'. Valid actions: navigate, screenshot, extract_text, extract_markdown, click, type, evaluate, expand, close",
                other
            )),
        }
    }

    async fn extract_research_markdown(&mut self, url: &str) -> Result<String, String> {
        let page = self.navigate(url).await?;
        self.extract_adaptive_markdown(&page).await
    }

    async fn page_for(&mut self, url: Option<&str>) -> Result<chromiumoxide::Page, String> {
        let page = match self.page.clone() {
            Some(page) => page,
            None => {
                let page = self
                    .browser
                    .new_page("about:blank")
                    .await
                    .map_err(|e| format!("Failed to open page: {}", e))?;
                self.page = Some(page.clone());
                page
            }
        };

        if let Some(url) = url {
            self.goto_if_needed(&page, url).await?;
        } else if self.current_url.is_none() {
            return Err(
                "This action requires 'url', or run action='navigate' first in this session"
                    .to_string(),
            );
        }

        Ok(page)
    }

    async fn navigate(&mut self, url: &str) -> Result<chromiumoxide::Page, String> {
        if url.trim().is_empty() {
            return Err("navigate requires non-empty 'url'".to_string());
        }
        let page = self.page_for(Some(url)).await?;
        Ok(page)
    }

    async fn goto_if_needed(
        &mut self,
        page: &chromiumoxide::Page,
        url: &str,
    ) -> Result<(), String> {
        if self.current_url.as_deref() == Some(url) {
            return Ok(());
        }
        page.goto(url)
            .await
            .map_err(|e| format!("Navigation failed: {}", e))?;
        self.wait_ready(page).await;
        self.current_url = Some(url.to_string());
        Ok(())
    }

    async fn wait_ready(&self, page: &chromiumoxide::Page) {
        let _ = page
            .evaluate(
                r#"(function(){
                    return new Promise(resolve => {
                        if (document.readyState === "complete" || document.readyState === "interactive") {
                            resolve(true);
                            return;
                        }
                        window.addEventListener("DOMContentLoaded", () => resolve(true), { once: true });
                        setTimeout(() => resolve(true), 1800);
                    });
                })()"#,
            )
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    async fn refresh_current_url(&mut self, page: &chromiumoxide::Page) {
        if let Ok(value) = page.evaluate("location.href").await {
            if let Ok(url) = value.into_value::<String>() {
                if !url.trim().is_empty() {
                    self.current_url = Some(url);
                }
            }
        }
    }

    async fn screenshot(&mut self, url: Option<&str>) -> Result<String, String> {
        use base64::Engine as _;
        use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
        use chromiumoxide::page::ScreenshotParams;

        let page = self.page_for(url).await?;
        let params = ScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .full_page(true)
            .build();
        let bytes: Vec<u8> = page
            .screenshot(params)
            .await
            .map_err(|e| format!("Screenshot failed: {}", e))?;
        self.refresh_current_url(&page).await;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Ok(format!("data:image/png;base64,{}", b64))
    }

    async fn click(&mut self, input: &BrowserInput) -> Result<String, String> {
        let page = self.page_for(input.url.as_deref()).await?;
        let selector = input
            .selector
            .as_ref()
            .ok_or_else(|| "click requires 'selector'".to_string())?;
        let selector_json = serde_json::to_string(selector)
            .map_err(|e| format!("selector encode failed: {}", e))?;
        let script = format!(
            "(function(){{ const el=document.querySelector({selector}); if(!el) return false; el.scrollIntoView({{block:'center', inline:'center'}}); el.click(); return true; }})()",
            selector = selector_json
        );

        let clicked: bool = page
            .evaluate(script.as_str())
            .await
            .map_err(|e| format!("click evaluate failed: {}", e))?
            .into_value()
            .map_err(|e| format!("click decode failed: {}", e))?;
        self.wait_ready(&page).await;
        self.refresh_current_url(&page).await;
        if clicked {
            Ok(format!("Clicked selector {}", selector))
        } else {
            Err(format!("No element found for selector {}", selector))
        }
    }

    async fn type_text(&mut self, input: &BrowserInput) -> Result<String, String> {
        let page = self.page_for(input.url.as_deref()).await?;
        let selector = input
            .selector
            .as_ref()
            .ok_or_else(|| "type requires 'selector'".to_string())?;
        let text = input
            .text
            .as_ref()
            .ok_or_else(|| "type requires 'text'".to_string())?;
        let selector_json = serde_json::to_string(selector)
            .map_err(|e| format!("selector encode failed: {}", e))?;
        let text_json =
            serde_json::to_string(text).map_err(|e| format!("text encode failed: {}", e))?;
        let script = format!(
            "(function(){{ const el=document.querySelector({selector}); if(!el) return false; el.focus(); el.value={text}; el.dispatchEvent(new Event('input', {{ bubbles: true }})); el.dispatchEvent(new Event('change', {{ bubbles: true }})); return true; }})()",
            selector = selector_json,
            text = text_json
        );

        let typed: bool = page
            .evaluate(script.as_str())
            .await
            .map_err(|e| format!("type evaluate failed: {}", e))?
            .into_value()
            .map_err(|e| format!("type decode failed: {}", e))?;
        self.refresh_current_url(&page).await;
        if typed {
            Ok(format!(
                "Filled selector {} with {} characters",
                selector,
                text.chars().count()
            ))
        } else {
            Err(format!("No element found for selector {}", selector))
        }
    }

    async fn evaluate(&mut self, input: &BrowserInput) -> Result<String, String> {
        let page = self.page_for(input.url.as_deref()).await?;
        let script = input
            .script
            .as_ref()
            .ok_or_else(|| "evaluate requires 'script'".to_string())?;
        let result = page
            .evaluate(script.as_str())
            .await
            .map_err(|e| format!("evaluate failed: {}", e))?;
        self.refresh_current_url(&page).await;
        match result.into_value::<Value>() {
            Ok(v) => Ok(v.to_string()),
            Err(e) => Ok(format!(
                "JavaScript executed, but result was not JSON-serializable: {}",
                e
            )),
        }
    }

    async fn extract_inner_text(&self, page: &chromiumoxide::Page) -> Result<String, String> {
        page.evaluate(
            "(function(){ return document && document.body ? document.body.innerText : ''; })()",
        )
        .await
        .map_err(|e| format!("extract_text evaluate failed: {}", e))?
        .into_value()
        .map_err(|e| format!("extract_text decode failed: {}", e))
    }

    async fn apply_recipes(&self, page: &chromiumoxide::Page) -> Result<String, String> {
        let value: Value = page
            .evaluate(BROWSER_RECIPES_SCRIPT)
            .await
            .map_err(|e| format!("browser recipes failed: {}", e))?
            .into_value()
            .map_err(|e| format!("browser recipes decode failed: {}", e))?;
        Ok(format!(
            "Applied browser recipes: opened_details={}, expanded_controls={}, dismissed_popups={}, visited_tabs={}",
            value
                .get("openedDetails")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            value
                .get("expandedControls")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            value
                .get("dismissedPopups")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            value
                .get("visitedTabs")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        ))
    }

    async fn extract_adaptive_markdown(
        &self,
        page: &chromiumoxide::Page,
    ) -> Result<String, String> {
        let value: Value = page
            .evaluate(ADAPTIVE_MARKDOWN_SCRIPT)
            .await
            .map_err(|e| format!("extract_markdown evaluate failed: {}", e))?
            .into_value()
            .map_err(|e| format!("extract_markdown decode failed: {}", e))?;

        let title = value
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)");
        let url = value
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("(current page)");
        let text = value.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let recipes = value.get("recipes").cloned().unwrap_or(Value::Null);
        let mut out = format!(
            "# {}\n\nSource: {}\nRecipes: {}\n\n## Page Text\n\n{}",
            title,
            url,
            recipes,
            truncate_browser_text(text, 80_000)
        );

        if let Some(blocks) = value.get("codeBlocks").and_then(|v| v.as_array()) {
            let code_blocks = blocks
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .take(20)
                .map(|block| format!("```text\n{}\n```", truncate_browser_text(block, 10_000)))
                .collect::<Vec<_>>();
            if !code_blocks.is_empty() {
                out.push_str("\n\n## Code Blocks\n\n");
                out.push_str(&code_blocks.join("\n\n"));
            }
        }
        if let Some(blocks) = value.get("structuredData").and_then(|v| v.as_array()) {
            let structured = blocks
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .take(8)
                .map(|block| format!("```json\n{}\n```", truncate_browser_text(block, 12_000)))
                .collect::<Vec<_>>();
            if !structured.is_empty() {
                out.push_str("\n\n## Structured Data\n\n");
                out.push_str(&structured.join("\n\n"));
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "browser")]
fn truncate_browser_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!(
            "{}\n\n... (truncated, {} total characters)",
            mangocode_core::truncate::truncate_bytes_prefix(text, max),
            text.len()
        )
    }
}

#[cfg(feature = "browser")]
const BROWSER_RECIPES_SCRIPT: &str = r#"
(function(){
  const visible = el => {
    if (!el || !el.isConnected) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  let openedDetails = 0, expandedControls = 0, dismissedPopups = 0, visitedTabs = 0;
  document.querySelectorAll("details:not([open])").forEach(el => { el.open = true; openedDetails++; });

  const closeSelectors = [
    "button[aria-label*='close' i]",
    "button[title*='close' i]",
    "[data-testid*='close' i]",
    "[class*='close' i]",
    "[aria-label*='dismiss' i]",
    "button[aria-label*='accept' i]",
    "button[title*='accept' i]",
    "button[class*='accept' i]",
    "button[class*='cookie' i]",
    "button:has(svg)"
  ];
  for (const selector of closeSelectors) {
    for (const el of Array.from(document.querySelectorAll(selector)).slice(0, 8)) {
      if (visible(el)) {
        try { el.click(); dismissedPopups++; } catch (_) {}
      }
    }
  }

  const expandSelectors = "[aria-expanded='false'], [data-state='closed'], button[class*='show' i], button[class*='more' i], button[class*='expand' i], summary";
  for (const el of Array.from(document.querySelectorAll(expandSelectors)).slice(0, 120)) {
    if (visible(el)) {
      try { el.click(); expandedControls++; } catch (_) {}
    }
  }

  for (const tab of Array.from(document.querySelectorAll("[role='tab'], [data-tab], button[class*='tab' i]")).slice(0, 50)) {
    if (visible(tab) && tab.getAttribute("aria-selected") !== "true") {
      try { tab.click(); visitedTabs++; } catch (_) {}
    }
  }

  return { openedDetails, expandedControls, dismissedPopups, visitedTabs };
})()
"#;

#[cfg(feature = "browser")]
const ADAPTIVE_MARKDOWN_SCRIPT: &str = r#"
(function(){
  const recipes = (function(){
    const visible = el => {
      if (!el || !el.isConnected) return false;
      const style = window.getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    };
    let openedDetails = 0, expandedControls = 0, dismissedPopups = 0, visitedTabs = 0;
    document.querySelectorAll("details:not([open])").forEach(el => { el.open = true; openedDetails++; });
    for (const selector of [
      "button[aria-label*='close' i]",
      "button[title*='close' i]",
      "[data-testid*='close' i]",
      "[aria-label*='dismiss' i]",
      "button[aria-label*='accept' i]",
      "button[title*='accept' i]",
      "button[class*='accept' i]",
      "button[class*='cookie' i]"
    ]) {
      for (const el of Array.from(document.querySelectorAll(selector)).slice(0, 8)) {
        if (visible(el)) { try { el.click(); dismissedPopups++; } catch (_) {} }
      }
    }
    const expandSelectors = "[aria-expanded='false'], [data-state='closed'], button[class*='show' i], button[class*='more' i], button[class*='expand' i], summary";
    for (const el of Array.from(document.querySelectorAll(expandSelectors)).slice(0, 120)) {
      if (visible(el)) { try { el.click(); expandedControls++; } catch (_) {} }
    }
    for (const tab of Array.from(document.querySelectorAll("[role='tab'], [data-tab], button[class*='tab' i]")).slice(0, 50)) {
      if (visible(tab) && tab.getAttribute("aria-selected") !== "true") {
        try { tab.click(); visitedTabs++; } catch (_) {}
      }
    }
    return { openedDetails, expandedControls, dismissedPopups, visitedTabs };
  })();

  const root = document.querySelector("main, article, [role='main'], .markdown-body, .theme-doc-markdown, .docs-content, .doc-content, .content, #content") || document.body;
  const codeBlocks = Array.from(document.querySelectorAll("pre, pre code, figure code"))
    .map(el => (el.innerText || el.textContent || "").trim())
    .filter(Boolean)
    .filter((value, idx, arr) => arr.indexOf(value) === idx)
    .slice(0, 30);
  const structuredData = Array.from(document.querySelectorAll("script[type='application/ld+json'], script#__NEXT_DATA__, script[id*='data' i]"))
    .map(el => (el.textContent || "").trim())
    .filter(Boolean)
    .slice(0, 8);
  return {
    title: document.title || "",
    url: location.href,
    text: root ? (root.innerText || root.textContent || "") : "",
    codeBlocks,
    structuredData,
    recipes
  };
})()
"#;
