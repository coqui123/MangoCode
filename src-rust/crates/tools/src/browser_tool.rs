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
        "Control a headless browser. Navigate to URLs, take screenshots, extract text, click elements, fill forms."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Dangerous
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": { "enum": ["navigate", "screenshot", "extract_text", "click", "type", "evaluate"] },
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
static LAST_URL_BY_SESSION: once_cell::sync::Lazy<dashmap::DashMap<String, String>> =
    once_cell::sync::Lazy::new(dashmap::DashMap::new);

#[cfg(feature = "browser")]
async fn execute_browser_action(params: BrowserInput, ctx: &ToolContext) -> ToolResult {
    use chromiumoxide::browser::{Browser, BrowserConfig};
    use futures::StreamExt;

    let config = match BrowserConfig::builder().build() {
        Ok(c) => c,
        Err(e) => return ToolResult::error(format!("Failed to build browser config: {}", e)),
    };

    let (browser, mut handler) = match Browser::launch(config).await {
        Ok(tuple) => tuple,
        Err(e) => {
            return ToolResult::error(format!(
                "Failed to launch Chromium. Ensure Chrome/Chromium is installed and discoverable: {}",
                e
            ))
        }
    };

    let handler_task = tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if event.is_err() {
                break;
            }
        }
    });

    let action_result = run_action_with_browser(&browser, &params, &ctx.session_id).await;

    let mut browser = browser;
    let _ = browser.close().await;
    let _ = browser.wait().await;
    handler_task.abort();

    match action_result {
        Ok(v) => ToolResult::success(v),
        Err(e) => ToolResult::error(e),
    }
}

#[cfg(feature = "browser")]
fn resolve_target_url(input: &BrowserInput, session_id: &str) -> Result<String, String> {
    if let Some(url) = &input.url {
        if url.trim().is_empty() {
            return Err("'url' cannot be empty".to_string());
        }
        return Ok(url.clone());
    }

    if let Some(saved) = LAST_URL_BY_SESSION.get(session_id) {
        return Ok(saved.value().clone());
    }

    Err("This action requires 'url', or run action='navigate' first in this session".to_string())
}

#[cfg(feature = "browser")]
async fn run_action_with_browser(
    browser: &chromiumoxide::browser::Browser,
    input: &BrowserInput,
    session_id: &str,
) -> Result<String, String> {
    use base64::Engine as _;
    use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
    use chromiumoxide::page::ScreenshotParams;

    match input.action.as_str() {
        "navigate" => {
            let url = input
                .url
                .as_ref()
                .ok_or_else(|| "navigate requires 'url'".to_string())?
                .clone();
            if url.trim().is_empty() {
                return Err("navigate requires non-empty 'url'".to_string());
            }

            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;
            let title = page
                .get_title()
                .await
                .map_err(|e| format!("Failed to read page title: {}", e))?
                .unwrap_or_else(|| "(untitled)".to_string());
            LAST_URL_BY_SESSION.insert(session_id.to_string(), url.clone());
            Ok(format!("Navigated to {} (title: {})", url, title))
        }
        "screenshot" => {
            let url = resolve_target_url(input, session_id)?;
            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;

            let params = ScreenshotParams::builder()
                .format(CaptureScreenshotFormat::Png)
                .full_page(true)
                .build();
            let bytes: Vec<u8> = page
                .screenshot(params)
                .await
                .map_err(|e| format!("Screenshot failed: {}", e))?;

            LAST_URL_BY_SESSION.insert(session_id.to_string(), url);
            let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
            Ok(format!("data:image/png;base64,{}", b64))
        }
        "extract_text" => {
            let url = resolve_target_url(input, session_id)?;
            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;

            let text: String = page
                .evaluate(
                    "(function(){ return document && document.body ? document.body.innerText : ''; })()",
                )
                .await
                .map_err(|e| format!("extract_text evaluate failed: {}", e))?
                .into_value()
                .map_err(|e| format!("extract_text decode failed: {}", e))?;

            LAST_URL_BY_SESSION.insert(session_id.to_string(), url);
            Ok(text)
        }
        "click" => {
            let url = resolve_target_url(input, session_id)?;
            let selector = input
                .selector
                .as_ref()
                .ok_or_else(|| "click requires 'selector'".to_string())?;
            let selector_json =
                serde_json::to_string(selector).map_err(|e| format!("selector encode failed: {}", e))?;

            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;

            let script = format!(
                "(function(){{ const el=document.querySelector({selector}); if(!el) return false; el.click(); return true; }})()",
                selector = selector_json
            );

            let clicked: bool = page
                .evaluate(script.as_str())
                .await
                .map_err(|e| format!("click evaluate failed: {}", e))?
                .into_value()
                .map_err(|e| format!("click decode failed: {}", e))?;

            LAST_URL_BY_SESSION.insert(session_id.to_string(), url);
            if clicked {
                Ok(format!("Clicked selector {}", selector))
            } else {
                Err(format!("No element found for selector {}", selector))
            }
        }
        "type" => {
            let url = resolve_target_url(input, session_id)?;
            let selector = input
                .selector
                .as_ref()
                .ok_or_else(|| "type requires 'selector'".to_string())?;
            let text = input
                .text
                .as_ref()
                .ok_or_else(|| "type requires 'text'".to_string())?;
            let selector_json =
                serde_json::to_string(selector).map_err(|e| format!("selector encode failed: {}", e))?;
            let text_json = serde_json::to_string(text).map_err(|e| format!("text encode failed: {}", e))?;

            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;

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

            LAST_URL_BY_SESSION.insert(session_id.to_string(), url);
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
        "evaluate" => {
            let url = resolve_target_url(input, session_id)?;
            let script = input
                .script
                .as_ref()
                .ok_or_else(|| "evaluate requires 'script'".to_string())?;

            let page = browser
                .new_page("about:blank")
                .await
                .map_err(|e| format!("Failed to open page: {}", e))?;
            page.goto(url.as_str())
                .await
                .map_err(|e| format!("Navigation failed: {}", e))?;

            let result = page
                .evaluate(script.as_str())
                .await
                .map_err(|e| format!("evaluate failed: {}", e))?;

            LAST_URL_BY_SESSION.insert(session_id.to_string(), url);

            match result.into_value::<Value>() {
                Ok(v) => Ok(v.to_string()),
                Err(e) => Ok(format!(
                    "JavaScript executed, but result was not JSON-serializable: {}",
                    e
                )),
            }
        }
        other => Err(format!(
            "Unknown action '{}'. Valid actions: navigate, screenshot, extract_text, click, type, evaluate",
            other
        )),
    }
}
