//! Chrome DevTools Protocol client for `/chrome` slash commands.

use mangocode_core::chrome_js;
use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::Message as WsMessage, MaybeTlsStream, WebSocketStream,
};

// ---------------------------------------------------------------------------
// Global session
// ---------------------------------------------------------------------------

static SESSION: Lazy<Mutex<Option<ChromeSession>>> = Lazy::new(|| Mutex::new(None));

static MSG_ID: Lazy<std::sync::atomic::AtomicU64> =
    Lazy::new(|| std::sync::atomic::AtomicU64::new(1));

fn next_id() -> u64 {
    MSG_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub struct ChromeSession {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    port: u16,
    tab_url: String,
    cdp_session_id: Option<String>,
    target_id: Option<String>,
    connection: ConnectionKind,
    event_buffer: VecDeque<Value>,
    pending_js_dialog: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnectionKind {
    BrowserFlatten,
    PageDirect,
}

fn take_session() -> anyhow::Result<ChromeSession> {
    SESSION.lock().take().ok_or_else(|| {
        anyhow::anyhow!("No active Chrome session. Run `/chrome connect` first.")
    })
}

fn store_session(s: ChromeSession) {
    *SESSION.lock() = Some(s);
}

// ---------------------------------------------------------------------------
// WS URL resolution
// ---------------------------------------------------------------------------

fn candidate_profile_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(h) = dirs::home_dir() {
        #[cfg(windows)]
        {
            out.push(h.join("AppData/Local/Google/Chrome/User Data"));
            out.push(h.join("AppData/Local/Google/Chrome SxS/User Data"));
            out.push(h.join("AppData/Local/Chromium/User Data"));
            out.push(h.join("AppData/Local/Microsoft/Edge/User Data"));
            out.push(h.join("AppData/Local/BraveSoftware/Brave-Browser/User Data"));
        }
        #[cfg(not(windows))]
        {
            out.push(h.join(".config/google-chrome"));
            out.push(h.join(".config/chromium"));
            out.push(h.join(".config/chromium-browser"));
            out.push(h.join(".config/microsoft-edge"));
        }
    }
    out
}

fn ws_from_devtools_active_port(http_base: &str) -> Option<String> {
    let base = http_base.trim_end_matches('/').replace("http://", "");
    let port_str = base.split(':').next_back()?.trim();
    if port_str.is_empty() {
        return None;
    }
    let host = if let Some(i) = base.rfind(':') {
        &base[..i]
    } else {
        return None;
    };
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    for root in candidate_profile_roots() {
        let active = root.join("DevToolsActivePort");
        let text = std::fs::read_to_string(active).ok()?;
        let mut lines = text.lines();
        let port = lines.next()?.trim();
        let ws_path = lines.next().unwrap_or("").trim();
        if port == port_str && !ws_path.is_empty() {
            return Some(format!("ws://{host}:{port}{ws_path}"));
        }
    }
    None
}

async fn fetch_json(client: &reqwest::Client, url: &str) -> anyhow::Result<Value> {
    let r = client.get(url).send().await?;
    if !r.status().is_success() {
        anyhow::bail!("HTTP {} for {}", r.status(), url);
    }
    Ok(r.json().await?)
}

pub async fn resolve_debugger_ws(port: u16) -> anyhow::Result<String> {
    if let Ok(ws) = std::env::var("MANGOCODE_CDP_WS") {
        let t = ws.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
    }

    if let Ok(u) = std::env::var("MANGOCODE_CDP_URL") {
        let base = u.trim_end_matches('/').to_string();
        if !base.is_empty() {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(8))
                .build()?;
            let ver_url = format!("{base}/json/version");
            match fetch_json(&client, &ver_url).await {
                Ok(v) => {
                    if let Some(ws) = v.get("webSocketDebuggerUrl").and_then(|x| x.as_str()) {
                        return Ok(ws.to_string());
                    }
                }
                Err(e) => {
                    let status = e
                        .downcast_ref::<reqwest::Error>()
                        .and_then(|re| re.status())
                        .map(|s| s.as_u16());
                    if status == Some(404) {
                        if let Some(ws) = ws_from_devtools_active_port(&base) {
                            return Ok(ws);
                        }
                    }
                }
            }
            if let Some(ws) = ws_from_devtools_active_port(&base) {
                return Ok(ws);
            }
            anyhow::bail!(
                "Could not resolve DevTools WebSocket from MANGOCODE_CDP_URL={base}. \
Try MANGOCODE_CDP_WS with a full ws:// URL from chrome://inspect."
            );
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()?;
    let base = format!("http://127.0.0.1:{port}");
    let ver_url = format!("{base}/json/version");
    match fetch_json(&client, &ver_url).await {
        Ok(v) => {
            if let Some(ws) = v.get("webSocketDebuggerUrl").and_then(|x| x.as_str()) {
                return Ok(ws.to_string());
            }
        }
        Err(e) => {
            let status = e
                .downcast_ref::<reqwest::Error>()
                .and_then(|re| re.status())
                .map(|s| s.as_u16());
            if status == Some(404) {
                if let Some(ws) = ws_from_devtools_active_port(&base) {
                    return Ok(ws);
                }
            }
        }
    }

    let list_url = format!("{base}/json/list");
    let tabs: Value = match fetch_json(&client, &list_url).await {
        Ok(v) => v,
        Err(_) => {
            if let Some(ws) = ws_from_devtools_active_port(&base) {
                return Ok(ws);
            }
            anyhow::bail!(
                "No DevTools endpoint on port {port}. Launch Chrome with --remote-debugging-port={port}"
            );
        }
    };

    if let Some(arr) = tabs.as_array() {
        if let Some(t) = arr.iter().find(|t| t["type"] == "browser") {
            if let Some(ws) = t.get("webSocketDebuggerUrl").and_then(|x| x.as_str()) {
                return Ok(ws.to_string());
            }
        }
        if let Some(t) = arr.iter().find(|t| t["type"] == "page") {
            if let Some(ws) = t.get("webSocketDebuggerUrl").and_then(|x| x.as_str()) {
                return Ok(ws.to_string());
            }
        }
    }

    anyhow::bail!(
        "No WebSocket debugger URL on port {}. Open a tab or enable remote debugging.",
        port
    )
}

const INTERNAL_PREFIX: &[&str] = &[
    "chrome://",
    "chrome-untrusted://",
    "devtools://",
    "chrome-extension://",
    "about:",
];

fn is_real_page_url(url: &str) -> bool {
    !INTERNAL_PREFIX.iter().any(|p| url.starts_with(p))
}

// ---------------------------------------------------------------------------
// CDP wire
// ---------------------------------------------------------------------------

fn session_param_for_method(method: &str, session_id: Option<&str>) -> Option<String> {
    if method.starts_with("Target.") {
        return None;
    }
    session_id.map(|s| s.to_string())
}

async fn read_cdp_message(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> anyhow::Result<Value> {
    loop {
        let raw = ws
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("WebSocket closed unexpectedly"))??;
        let text: String = match raw {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Ping(p) => {
                ws.send(WsMessage::Pong(p)).await?;
                continue;
            }
            WsMessage::Pong(_) => continue,
            WsMessage::Close(_) => anyhow::bail!("WebSocket closed by Chrome"),
            _ => continue,
        };
        return Ok(serde_json::from_str(&text)?);
    }
}

fn handle_cdp_event(session: &mut ChromeSession, msg: &Value) {
    let Some(method) = msg.get("method").and_then(|m| m.as_str()) else {
        return;
    };
    match method {
        "Page.javascriptDialogOpening" => {
            session.pending_js_dialog = Some(msg.get("params").cloned().unwrap_or(Value::Null));
        }
        "Page.javascriptDialogClosed" => {
            session.pending_js_dialog = None;
        }
        _ => {}
    }
}

async fn cdp_call(
    session: &mut ChromeSession,
    method: &str,
    params: Value,
) -> anyhow::Result<Value> {
    let id = next_id();
    let mut req = json!({
        "id": id,
        "method": method,
        "params": params,
    });
    if let Some(sid) = session_param_for_method(method, session.cdp_session_id.as_deref()) {
        req["sessionId"] = json!(sid);
    }

    session
        .ws
        .send(WsMessage::Text(req.to_string()))
        .await?;

    loop {
        while let Some(front) = session.event_buffer.pop_front() {
            if front.get("id") == Some(&json!(id)) {
                if let Some(err) = front.get("error") {
                    anyhow::bail!("CDP error: {}", err);
                }
                return Ok(front);
            }
            handle_cdp_event(session, &front);
        }

        let msg = read_cdp_message(&mut session.ws).await?;
        if msg.get("id") == Some(&json!(id)) {
            if let Some(err) = msg.get("error") {
                anyhow::bail!("CDP error: {}", err);
            }
            return Ok(msg);
        }
        handle_cdp_event(session, &msg);
        session.event_buffer.push_back(msg);
    }
}

async fn attach_first_real_page(session: &mut ChromeSession) -> anyhow::Result<()> {
    let resp = cdp_call(session, "Target.getTargets", json!({})).await?;
    let targets = resp["result"]["targetInfos"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Target.getTargets: no targetInfos"))?;

    let page = targets
        .iter()
        .find(|t| {
            t.get("type").and_then(|x| x.as_str()) == Some("page")
                && t.get("url")
                    .and_then(|x| x.as_str())
                    .map(is_real_page_url)
                    .unwrap_or(false)
        })
        .or_else(|| targets.iter().find(|t| t["type"] == "page"))
        .ok_or_else(|| anyhow::anyhow!("No page target to attach"))?;

    let target_id = page["targetId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("page target without targetId"))?
        .to_string();
    let tab_url = page
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("")
        .to_string();

    let attach = cdp_call(
        session,
        "Target.attachToTarget",
        json!({ "targetId": &target_id, "flatten": true }),
    )
    .await?;

    let sid = attach["result"]["sessionId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("attachToTarget missing sessionId"))?
        .to_string();

    session.cdp_session_id = Some(sid);
    session.target_id = Some(target_id);
    session.tab_url = tab_url;
    session.connection = ConnectionKind::BrowserFlatten;

    enable_domains_on_session(session).await?;
    Ok(())
}

async fn enable_domains_on_session(session: &mut ChromeSession) -> anyhow::Result<()> {
    let _ = cdp_call(session, "Page.enable", json!({})).await?;
    let _ = cdp_call(session, "Runtime.enable", json!({})).await?;
    let _ = cdp_call(session, "Network.enable", json!({})).await?;
    let _ = cdp_call(session, "DOM.enable", json!({})).await?;
    Ok(())
}

async fn open_page_session(
    ws_url: &str,
    port: u16,
    tab_url: String,
) -> anyhow::Result<ChromeSession> {
    let (ws, _) = connect_async(ws_url)
        .await
        .map_err(|e| anyhow::anyhow!("WebSocket connect to {} failed: {}", ws_url, e))?;

    let mut session = ChromeSession {
        ws,
        port,
        tab_url,
        cdp_session_id: None,
        target_id: None,
        connection: ConnectionKind::PageDirect,
        event_buffer: VecDeque::with_capacity(64),
        pending_js_dialog: None,
    };
    let _ = cdp_call(&mut session, "Page.enable", json!({})).await?;
    let _ = cdp_call(&mut session, "Runtime.enable", json!({})).await?;
    let _ = cdp_call(&mut session, "Network.enable", json!({})).await?;
    Ok(session)
}

pub async fn connect(port: u16) -> anyhow::Result<String> {
    let ws_url = resolve_debugger_ws(port).await?;

    let (ws, _) = connect_async(&ws_url)
        .await
        .map_err(|e| anyhow::anyhow!("WebSocket connect failed: {}", e))?;

    let mut session = ChromeSession {
        ws,
        port,
        tab_url: String::new(),
        cdp_session_id: None,
        target_id: None,
        connection: ConnectionKind::PageDirect,
        event_buffer: VecDeque::with_capacity(64),
        pending_js_dialog: None,
    };

    let can_target = cdp_call(&mut session, "Target.getTargets", json!({}))
        .await
        .is_ok();

    if can_target {
        match attach_first_real_page(&mut session).await {
            Ok(()) => {
                let tab = session.tab_url.clone();
                store_session(session);
                return Ok(format!(
                    "Connected (browser DevTools, flattened session) on port {port} - tab: {tab}"
                ));
            }
            Err(e) => {
                tracing::warn!("Target attach failed, falling back to page target: {}", e);
            }
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()?;
    let list_url = format!("http://127.0.0.1:{port}/json/list");
    let tabs: Value = fetch_json(&client, &list_url)
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Could not connect or enumerate tabs. Try MANGOCODE_CDP_WS with a ws:// URL from chrome://inspect."
            )
        })?;

    let first_page = tabs
        .as_array()
        .and_then(|a| a.iter().find(|t| t.get("type").and_then(|x| x.as_str()) == Some("page")));

    let t = first_page.ok_or_else(|| anyhow::anyhow!("No page tab found on port {}", port))?;
    let page_ws = t
        .get("webSocketDebuggerUrl")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow::anyhow!("page target missing webSocketDebuggerUrl"))?;
    let tab_url = t
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("")
        .to_string();

    let session = open_page_session(page_ws, port, tab_url.clone()).await?;
    store_session(session);
    Ok(format!(
        "Connected (page target) on port {port} - {tab_url}"
    ))
}

pub fn disconnect() -> String {
    let mut guard = SESSION.lock();
    if let Some(session) = guard.take() {
        format!(
            "Disconnected from Chrome on port {} (tab: {}).",
            session.port, session.tab_url
        )
    } else {
        "No active Chrome session.".to_string()
    }
}

pub async fn navigate(url: &str) -> anyhow::Result<String> {
    let url = url.to_string();
    let mut s = take_session()?;
    let result = async {
        let resp = cdp_call(&mut s, "Page.navigate", json!({ "url": url })).await?;
        let frame_id = resp["result"]["frameId"].as_str().unwrap_or("unknown");
        Ok(format!("Navigated. frameId={}", frame_id))
    }
    .await;
    store_session(s);
    result
}

pub async fn screenshot() -> anyhow::Result<String> {
    let mut s = take_session()?;
    let result = async {
        let resp = cdp_call(
            &mut s,
            "Page.captureScreenshot",
            json!({ "format": "png", "captureBeyondViewport": false }),
        )
        .await?;
        let b64 = resp["result"]["data"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("No screenshot data in response"))?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;

        let tmp = tempfile::Builder::new()
            .prefix("cc-chrome-")
            .suffix(".png")
            .tempfile()?;
        let path = tmp.path().to_path_buf();
        std::fs::write(&path, &bytes)?;
        let _ = tmp.keep()?;
        Ok(format!("Screenshot saved to {}", path.display()))
    }
    .await;
    store_session(s);
    result
}

pub async fn click(selector: &str) -> anyhow::Result<String> {
    let sel_json = serde_json::to_string(selector)?;
    let js = format!(
        r#"(function(){{
                var el=document.querySelector({sel});
                if(!el)return 'ELEMENT_NOT_FOUND';
                var r=el.getBoundingClientRect();
                return JSON.stringify({{x:r.left+r.width/2,y:r.top+r.height/2}});
            }})()"#,
        sel = sel_json
    );
    let selector = selector.to_string();
    let mut s = take_session()?;
    let result = async {
        let expr = chrome_js::prepare_eval_expression(&js);
        let resp = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({ "expression": expr, "returnByValue": true, "awaitPromise": true }),
        )
        .await?;
        let full = &resp["result"];
        let val_str = chrome_js::format_evaluate_response(full, &expr).map_err(anyhow::Error::msg)?;
        if val_str == "ELEMENT_NOT_FOUND" {
            anyhow::bail!("No element found for selector: {}", selector);
        }
        let coords: Value = serde_json::from_str(&val_str)?;
        let x = coords["x"].as_f64().unwrap_or(0.0);
        let y = coords["y"].as_f64().unwrap_or(0.0);

        cdp_call(
            &mut s,
            "Input.dispatchMouseEvent",
            json!({
                "type": "mousePressed", "x": x, "y": y,
                "button": "left", "clickCount": 1
            }),
        )
        .await?;
        cdp_call(
            &mut s,
            "Input.dispatchMouseEvent",
            json!({
                "type": "mouseReleased", "x": x, "y": y,
                "button": "left", "clickCount": 1
            }),
        )
        .await?;

        Ok(format!("Clicked '{}' at ({:.0}, {:.0})", selector, x, y))
    }
    .await;
    store_session(s);
    result
}

pub async fn fill(selector: &str, text: &str) -> anyhow::Result<String> {
    let js = format!(
        r#"(function(){{
                var el=document.querySelector({sel});
                if(!el)return false;
                el.focus();
                el.value={val};
                el.dispatchEvent(new Event('input',{{bubbles:true}}));
                el.dispatchEvent(new Event('change',{{bubbles:true}}));
                return true;
            }})()"#,
        sel = serde_json::to_string(selector)?,
        val = serde_json::to_string(text)?
    );
    let selector = selector.to_string();
    let text = text.to_string();
    let mut s = take_session()?;
    let result = async {
        let expr = chrome_js::prepare_eval_expression(&js);
        let resp = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({ "expression": expr, "returnByValue": true, "awaitPromise": true }),
        )
        .await?;
        let ok = resp["result"]["result"].get("value") == Some(&json!(true))
            || matches!(
                chrome_js::format_evaluate_response(&resp["result"], &expr),
                Ok(ref s) if s == "true"
            );
        if ok {
            Ok(format!("Filled '{}' with {:?}", selector, text))
        } else {
            Err(anyhow::anyhow!(
                "No element found for selector: {}",
                selector
            ))
        }
    }
    .await;
    store_session(s);
    result
}

fn merge_key_event(base: &Value, typ: &str, include_text: bool) -> Value {
    let mut o = base.as_object().unwrap().clone();
    o.insert("type".to_string(), json!(typ));
    if !include_text {
        o.remove("text");
    }
    Value::Object(o)
}

/// Real keystrokes for React/Vue-style inputs (browser-harness `fill_input` style).
pub async fn fill_keystrokes(selector: &str, text: &str, clear_first: bool) -> anyhow::Result<String> {
    let sel = serde_json::to_string(selector)?;
    let mut s = take_session()?;
    let result = async {
        let focus_js = format!(
            "(()=>{{const e=document.querySelector({});if(!e)return false;e.focus();return true;}})()",
            sel
        );
        let expr = chrome_js::prepare_eval_expression(&focus_js);
        let resp = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({ "expression": expr, "returnByValue": true, "awaitPromise": true }),
        )
        .await?;
        let ok = resp["result"]["result"].get("value") == Some(&json!(true));
        if !ok {
            anyhow::bail!("type_keystrokes: element not found for selector {:?}", selector);
        }

        let mods: i64 = if cfg!(target_os = "macos") { 4 } else { 2 };
        if clear_first {
            let sel_all = json!({
                "key": "a",
                "code": "KeyA",
                "modifiers": mods,
                "windowsVirtualKeyCode": 65i64,
                "nativeVirtualKeyCode": 65i64,
                "text": ""
            });
            cdp_call(
                &mut s,
                "Input.dispatchKeyEvent",
                merge_key_event(&sel_all, "rawKeyDown", false),
            )
            .await?;
            cdp_call(
                &mut s,
                "Input.dispatchKeyEvent",
                merge_key_event(&sel_all, "keyUp", false),
            )
            .await?;
            let bk = json!({
                "key": "Backspace",
                "code": "Backspace",
                "modifiers": 0,
                "windowsVirtualKeyCode": 8i64,
                "nativeVirtualKeyCode": 8i64,
                "text": ""
            });
            cdp_call(
                &mut s,
                "Input.dispatchKeyEvent",
                merge_key_event(&bk, "keyDown", false),
            )
            .await?;
            cdp_call(
                &mut s,
                "Input.dispatchKeyEvent",
                merge_key_event(&bk, "keyUp", false),
            )
            .await?;
        }

        for ch in text.chars() {
            if ch == '\n' || ch == '\r' {
                let ent = json!({
                    "key": "Enter",
                    "code": "Enter",
                    "modifiers": 0,
                    "windowsVirtualKeyCode": 13i64,
                    "nativeVirtualKeyCode": 13i64,
                    "text": ""
                });
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&ent, "keyDown", false),
                )
                .await?;
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&ent, "keyUp", false),
                )
                .await?;
            } else if ch == '\t' {
                let tab_ev = json!({
                    "key": "Tab",
                    "code": "Tab",
                    "modifiers": 0,
                    "windowsVirtualKeyCode": 9i64,
                    "nativeVirtualKeyCode": 9i64,
                    "text": ""
                });
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&tab_ev, "keyDown", false),
                )
                .await?;
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&tab_ev, "keyUp", false),
                )
                .await?;
            } else if !ch.is_control() && ch.is_ascii() {
                let upper = ch.to_ascii_uppercase();
                let code = if ch.is_ascii_alphabetic() {
                    format!("Key{upper}")
                } else if ch.is_ascii_digit() {
                    format!("Digit{ch}")
                } else {
                    String::new()
                };
                let vk = ch as u32 as i64;
                let key_s = ch.to_string();
                let ev = json!({
                    "key": key_s,
                    "code": code,
                    "modifiers": 0,
                    "windowsVirtualKeyCode": vk,
                    "nativeVirtualKeyCode": vk,
                    "text": key_s,
                });
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&ev, "keyDown", false),
                )
                .await?;
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&ev, "char", true),
                )
                .await?;
                cdp_call(
                    &mut s,
                    "Input.dispatchKeyEvent",
                    merge_key_event(&ev, "keyUp", false),
                )
                .await?;
            }
        }

        let dispatch_js = format!(
            "(()=>{{const e=document.querySelector({});if(!e)return;e.dispatchEvent(new Event('input',{{bubbles:true}}));e.dispatchEvent(new Event('change',{{bubbles:true}}));}})()",
            sel
        );
        let dex = chrome_js::prepare_eval_expression(&dispatch_js);
        let _ = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({ "expression": dex, "returnByValue": true, "awaitPromise": true }),
        )
        .await?;

        Ok(format!(
            "type_keystrokes: filled {:?} ({} chars)",
            selector,
            text.chars().count()
        ))
    }
    .await;
    store_session(s);
    result
}

pub async fn eval_js(user_js: &str) -> anyhow::Result<String> {
    let prepared = chrome_js::prepare_eval_expression(user_js);
    let mut s = take_session()?;
    let result = async {
        let resp = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({
                "expression": prepared,
                "returnByValue": true,
                "awaitPromise": true
            }),
        )
        .await?;
        chrome_js::format_evaluate_response(&resp["result"], &prepared).map_err(anyhow::Error::msg)
    }
    .await;
    store_session(s);
    result
}

pub async fn eval_in_iframe(url_substr: &str, user_js: &str) -> anyhow::Result<String> {
    let prepared = chrome_js::prepare_eval_expression(user_js);
    let mut s = take_session()?;
    let result = async {
        if s.connection != ConnectionKind::BrowserFlatten {
            anyhow::bail!(
                "iframe eval requires a browser-level DevTools connection (flattened session). \
Use `/chrome connect` with Chrome exposing Target.*; or set MANGOCODE_CDP_WS from chrome://inspect."
            );
        }

        let resp = cdp_call(&mut s, "Target.getTargets", json!({})).await?;
        let targets = resp["result"]["targetInfos"].as_array().ok_or_else(|| {
            anyhow::anyhow!("iframe: no targets")
        })?;

        let iframe = targets
            .iter()
            .find(|t| {
                t.get("type").and_then(|x| x.as_str()) == Some("iframe")
                    && t.get("url")
                        .and_then(|u| u.as_str())
                        .map(|u| u.contains(url_substr))
                        .unwrap_or(false)
            })
            .ok_or_else(|| {
                anyhow::anyhow!("No iframe target with URL containing {:?}", url_substr)
            })?;

        let iframe_tid = iframe["targetId"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("iframe without targetId"))?;

        let attach = cdp_call(
            &mut s,
            "Target.attachToTarget",
            json!({ "targetId": iframe_tid, "flatten": true }),
        )
        .await?;
        let iframe_sid = attach["result"]["sessionId"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("iframe attach missing sessionId"))?;

        let old_sid = s.cdp_session_id.replace(iframe_sid.to_string());
        let _ = cdp_call(&mut s, "Runtime.enable", json!({})).await?;

        let eval_result = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({
                "expression": prepared,
                "returnByValue": true,
                "awaitPromise": true
            }),
        )
        .await;

        s.cdp_session_id = old_sid;
        let _ = cdp_call(
            &mut s,
            "Target.detachFromTarget",
            json!({ "sessionId": iframe_sid }),
        )
        .await;

        let eval_resp = eval_result?;
        chrome_js::format_evaluate_response(&eval_resp["result"], &prepared)
            .map_err(anyhow::Error::msg)
    }
    .await;
    store_session(s);
    result
}

pub async fn tabs_list() -> anyhow::Result<String> {
    let port = {
        let g = SESSION.lock();
        let s = g
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No active session"))?;
        s.port
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()?;
    let tabs: Value = fetch_json(
        &client,
        &format!("http://127.0.0.1:{port}/json/list"),
    )
    .await?;

    let mut out = String::from("Tabs (from /json/list):\n");
    if let Some(arr) = tabs.as_array() {
        for t in arr {
            if t.get("type").and_then(|x| x.as_str()) != Some("page") {
                continue;
            }
            let id = t.get("targetId").and_then(|x| x.as_str()).unwrap_or("?");
            let title = t.get("title").and_then(|x| x.as_str()).unwrap_or("");
            let url = t.get("url").and_then(|x| x.as_str()).unwrap_or("");
            out.push_str(&format!("  {id} | {title} | {url}\n"));
        }
    }
    Ok(out)
}

pub async fn switch_tab(target_id: &str) -> anyhow::Result<String> {
    let mut s = take_session()?;
    if s.connection == ConnectionKind::BrowserFlatten {
        let r: anyhow::Result<String> = async {
            let _ = cdp_call(
                &mut s,
                "Target.activateTarget",
                json!({ "targetId": target_id }),
            )
            .await?;

            let attach = cdp_call(
                &mut s,
                "Target.attachToTarget",
                json!({ "targetId": target_id, "flatten": true }),
            )
            .await?;

            let sid = attach["result"]["sessionId"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("attach missing sessionId"))?
                .to_string();

            s.cdp_session_id = Some(sid);
            s.target_id = Some(target_id.to_string());

            if let Ok(info) = cdp_call(
                &mut s,
                "Target.getTargetInfo",
                json!({ "targetId": target_id }),
            )
            .await
            {
                s.tab_url = info["result"]["targetInfo"]["url"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
            }

            enable_domains_on_session(&mut s).await?;
            Ok(format!(
                "Switched active tab to target {} ({})",
                target_id, s.tab_url
            ))
        }
        .await;
        match r {
            Ok(msg) => {
                store_session(s);
                Ok(msg)
            }
            Err(e) => {
                store_session(s);
                Err(e)
            }
        }
    } else {
        let port = s.port;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(8))
            .build()?;
        let tabs: Value = match fetch_json(
            &client,
            &format!("http://127.0.0.1:{port}/json/list"),
        )
        .await
        {
            Ok(t) => t,
            Err(e) => {
                store_session(s);
                return Err(e);
            }
        };

        let entry = tabs.as_array().and_then(|a| {
            a.iter()
                .find(|t| t.get("targetId").and_then(|x| x.as_str()) == Some(target_id))
        });

        let Some(entry) = entry else {
            store_session(s);
            anyhow::bail!("targetId not found in /json/list");
        };

        let ws = entry
            .get("webSocketDebuggerUrl")
            .and_then(|x| x.as_str())
            .ok_or_else(|| anyhow::anyhow!("no webSocketDebuggerUrl for target"));
        let ws = match ws {
            Ok(w) => w,
            Err(e) => {
                store_session(s);
                return Err(e);
            }
        };
        let tab_url = entry
            .get("url")
            .and_then(|u| u.as_str())
            .unwrap_or("")
            .to_string();
        drop(s);
        let session = open_page_session(ws, port, tab_url.clone()).await?;
        store_session(session);
        Ok(format!(
            "Reconnected to page target {} - {tab_url} (page-direct mode)",
            target_id
        ))
    }
}

pub async fn new_tab(url: Option<&str>) -> anyhow::Result<String> {
    let mut s = take_session()?;
    if s.connection != ConnectionKind::BrowserFlatten {
        store_session(s);
        anyhow::bail!(
            "new_tab requires browser-level DevTools (flattened session). \
Use MANGOCODE_CDP_WS from chrome://inspect or connect when Target.* works."
        );
    }
    let result: anyhow::Result<String> = async {
        let create = cdp_call(
            &mut s,
            "Target.createTarget",
            json!({ "url": "about:blank" }),
        )
        .await?;
        let tid = create["result"]["targetId"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("createTarget missing targetId"))?;

        let _ = cdp_call(&mut s, "Target.activateTarget", json!({ "targetId": tid })).await?;

        let attach = cdp_call(
            &mut s,
            "Target.attachToTarget",
            json!({ "targetId": tid, "flatten": true }),
        )
        .await?;
        let sid = attach["result"]["sessionId"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("attach missing sessionId"))?
            .to_string();

        s.cdp_session_id = Some(sid);
        s.target_id = Some(tid.to_string());
        s.tab_url = "about:blank".to_string();
        enable_domains_on_session(&mut s).await?;

        if let Some(u) = url {
            if !u.trim().is_empty() {
                let _ = cdp_call(&mut s, "Page.navigate", json!({ "url": u.trim() })).await?;
                s.tab_url = u.trim().to_string();
            }
        }
        Ok(format!("New tab target {tid} - {}", s.tab_url))
    }
    .await;

    match result {
        Ok(m) => {
            store_session(s);
            Ok(m)
        }
        Err(e) => {
            store_session(s);
            Err(e)
        }
    }
}

pub async fn page_info() -> anyhow::Result<String> {
    let mut s = take_session()?;
    let result = async {
        if let Some(d) = &s.pending_js_dialog {
            return Ok(format!("pending javascript dialog: {}", d));
        }
        let expr = "JSON.stringify({url:location.href,title:document.title,w:innerWidth,h:innerHeight,sx:scrollX,sy:scrollY,pw:document.documentElement.scrollWidth,ph:document.documentElement.scrollHeight})";
        let resp = cdp_call(
            &mut s,
            "Runtime.evaluate",
            json!({
                "expression": expr,
                "returnByValue": true,
                "awaitPromise": true
            }),
        )
        .await?;
        let text = chrome_js::format_evaluate_response(&resp["result"], expr)
            .map_err(anyhow::Error::msg)?;
        Ok(text)
    }
    .await;
    store_session(s);
    result
}

pub async fn handle_js_dialog(accept: bool, prompt_text: Option<&str>) -> anyhow::Result<String> {
    let mut s = take_session()?;
    let result = async {
        let mut p = json!({ "accept": accept });
        if let Some(t) = prompt_text {
            p.as_object_mut()
                .unwrap()
                .insert("promptText".to_string(), json!(t));
        }
        let _ = cdp_call(&mut s, "Page.handleJavaScriptDialog", p).await?;
        Ok(format!(
            "handle_javascript_dialog accept={} (prompt set: {})",
            accept,
            prompt_text.is_some()
        ))
    }
    .await;
    store_session(s);
    result
}

fn network_event_for_session(msg: &Value, session_id: Option<&str>) -> bool {
    msg.get("sessionId")
        .and_then(|v| v.as_str())
        .map(|sid| Some(sid) == session_id)
        .unwrap_or(session_id.is_none())
}

pub async fn wait_network_idle(timeout_secs: f64, idle_ms: u64) -> anyhow::Result<String> {
    let mut s = take_session()?;
    let result = async {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs);
        let sid = s.cdp_session_id.clone();
        let mut inflight: HashSet<String> = HashSet::new();
        let mut last_activity = Instant::now();

        fn process_network_event(
            inflight: &mut HashSet<String>,
            last_activity: &mut Instant,
            method: &str,
            params: &Value,
        ) {
            let rid = params
                .get("requestId")
                .and_then(|v| v.as_str())
                .map(|x| x.to_string());
            let Some(rid) = rid else {
                return;
            };
            match method {
                "Network.requestWillBeSent" => {
                    inflight.insert(rid);
                }
                "Network.loadingFinished" | "Network.loadingFailed" => {
                    inflight.remove(&rid);
                }
                _ => {}
            }
            *last_activity = Instant::now();
        }

        while let Some(ev) = s.event_buffer.pop_front() {
            if network_event_for_session(&ev, sid.as_deref())
                && ev.get("method").and_then(|m| m.as_str()).is_some()
            {
                let method = ev["method"].as_str().unwrap_or("");
                if let Some(params) = ev.get("params") {
                    process_network_event(&mut inflight, &mut last_activity, method, params);
                }
            }
        }

        while Instant::now() < deadline {
            if inflight.is_empty()
                && last_activity.elapsed().as_millis() as u64 >= idle_ms
            {
                store_session(s);
                return Ok(format!(
                    "network idle (no in-flight requests for {idle_ms} ms within {} s budget)",
                    timeout_secs
                ));
            }

            match tokio::time::timeout(Duration::from_millis(100), read_cdp_message(&mut s.ws))
                .await
            {
                Ok(Ok(msg)) => {
                    if msg.get("id").is_some() {
                        s.event_buffer.push_back(msg);
                        continue;
                    }
                    if !network_event_for_session(&msg, sid.as_deref()) {
                        continue;
                    }
                    if let Some(method) = msg.get("method").and_then(|m| m.as_str()) {
                        if let Some(params) = msg.get("params") {
                            if method.starts_with("Network.") {
                                process_network_event(
                                    &mut inflight,
                                    &mut last_activity,
                                    method,
                                    params,
                                );
                            }
                        }
                    }
                }
                Ok(Err(e)) => {
                    store_session(s);
                    return Err(e);
                }
                Err(_) => {
                    // idle poll window
                    if inflight.is_empty()
                        && last_activity.elapsed().as_millis() as u64 >= idle_ms
                    {
                        store_session(s);
                        return Ok(format!(
                            "network idle (no in-flight requests for {idle_ms} ms)"
                        ));
                    }
                }
            }
        }
        store_session(s);
        anyhow::bail!("wait_network_idle timed out after {} s", timeout_secs);
    }
    .await;

    result
}
