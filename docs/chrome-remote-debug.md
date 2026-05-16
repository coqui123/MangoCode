# Chrome remote debugging and `/chrome`

MangoCode can drive a real Chrome/Chromium instance through the DevTools Protocol via the `/chrome` slash commands in **mangocode-commands** (`chrome_cdp.rs`), with shared eval string helpers in **mangocode-core** (`chrome_js.rs`).

## One-time browser setup

1. Launch Chrome with a debugging port, for example:

   `chrome --remote-debugging-port=9222 --no-first-run`

2. On recent Chrome versions, allow remote debugging when prompted, or use **chrome://inspect -> Configure...** and enable the port. If HTTP discovery (`/json/list`, `/json/version`) returns **404** on the default profile, MangoCode falls back to reading **`DevToolsActivePort`** under your Chrome user-data directory (same idea as [browser-harness](https://github.com/browser-use/browser-harness) `daemon.get_ws_url`).

## Environment overrides

| Variable | Purpose |
|----------|---------|
| `MANGOCODE_CDP_WS` | Full WebSocket URL to the DevTools endpoint (copy from **chrome://inspect** -> *Open dedicated DevTools for ...* / target link). Highest priority. |
| `MANGOCODE_CDP_URL` | HTTP base only (e.g. `http://127.0.0.1:9333`). The client calls `/json/version`; on **404**, it tries **`DevToolsActivePort`** next. |

## Command recap

Shared **`mangocode_core::chrome_js`** helpers implement the same **return / IIFE** behavior for **`/chrome eval`** and the **`Browser`** tool's **`evaluate`** action.
- **`/chrome typekeys`** sends real **`Input.dispatchKeyEvent`** sequences (React/Vue-friendly), similar to browser-harness **`fill_input`**.
- **`/chrome tabs`** / **`/chrome tab`** list or switch targets; **`/chrome newtab`** creates a tab when connected in **browser flatten** mode.
- **`/chrome iframe_eval`**, **`/chrome page_info`**, **`/chrome dialog`**, **`/chrome wait_network`** cover iframe evaluation, JS dialogs, and network-idle waits.

## Further reading

The [browser-harness](https://github.com/browser-use/browser-harness) repo's **`interaction-skills/`** markdown files describe cross-browser patterns (shadow DOM, dialogs, iframes) that apply to any CDP client; MangoCode does not ship that tree, but the same techniques work with **`/chrome`**.

For headless extraction and bot-wall heuristics, see the **`Browser`** tool in `mangocode-tools` (`browser_tool.rs`).
