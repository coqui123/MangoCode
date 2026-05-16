# MangoCode Intelligence Features

This page covers the native features MangoCode uses to stay accurate without depending on extra agent apps: Smart Attachments, Mango Research, output compression, and layered memory.

## Smart Attachments

Paste or attach files in the TUI prompt. MangoCode routes each attachment by file type and selected model capability.

- Images are sent as raw visual `ContentBlock::Image` when the selected provider/model supports image input.
- Images are converted to Markdown metadata plus local OCR text when the model does not support images and OCR is available.
- PDFs are converted to Markdown by default because text Markdown is usually more useful and token-efficient for coding than raw PDF passthrough.
- DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, ZIP, and text-like files use MangoCode's native Markdown extraction.
- Extracted Markdown is cached under `~/.mangocode/attachments`.

### Local OCR With Tesseract

MangoCode can OCR scanned PDFs and non-vision image fallbacks with a local Tesseract executable. This is not a cloud OCR API. For image-heavy PDFs, MangoCode first tries Tesseract directly and then falls back to extracting embedded JPEG page images for OCR.

Path resolution order:

1. `attachments.tesseract_path` in MangoCode settings.
2. `MANGOCODE_TESSERACT_PATH`.
3. Vault key `tesseract` or `ocr-tesseract`.
4. `tesseract` on `PATH`.

Example settings:

```toml
[attachments]
markdown_extraction_enabled = true
images_raw_by_default = true
ocr_enabled = true
tesseract_path = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
tesseract_lang = "eng"
```

Vault option:

```text
/vault set tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Environment option:

```powershell
$env:MANGOCODE_TESSERACT_PATH = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Mango Research

Mango Research is a coding-oriented docs pipeline:

```text
search -> rank -> fetch -> extract -> quality score -> rendered fallback -> cite -> cache -> index
```

Tools:

- `WebSearch` performs normal public web search without a search API key by using MangoCode's built-in DuckDuckGo instant-answer plus HTML/lite result parser. `BRAVE_SEARCH_API_KEY` is optional for users who want Brave as a higher-quality provider, not required for basic web search. Public web search still needs network access; the local research index works from cache when offline.
- `DocSearch` finds official or primary docs first and also searches MangoCode's local research index.
- `DocRead` reads one URL or local doc into clean Markdown with citation metadata.
- `DeepRead` reads multiple sources and returns a concise research brief with facts, source list, staleness notes, examples, and implementation implications.
- `WebFetch` fetches any public HTTP(S) URL to text/Markdown. `extract`=`auto`|`main` uses the research-style pipeline; **`extract`=`raw`** returns the response body—set **`rendered_fallback`:** **`true`** when raw HTML looks like a CDN/bot gate (or use **`auto`**/ **`main`**) so MangoCode can use browser extraction instead of silently caching useless challenge pages.
- In `full-tools` or explicit `tool-rendered-fetch` builds, `RenderedFetch` forces the browser-backed rendered extraction path directly when a docs page is heavily client-rendered or hidden behind tabs/accordions.

The source index lives at `~/.mangocode/research/research-v1/source-index.json`. Every successful `DocRead` or fetched research document is indexed, so future searches can reuse trusted docs even when web search is weak or offline.

### CDN / bot challenge pages

Some sites return short HTML challenges (often Cloudflare) to programmatic HTTP clients. MangoCode detects **challenge-shaped** snippets (markers like “checking your browser”, embedded challenge scripts, Turnstiles, compact Cloudflare error pages), and when relevant uses **HTTP status** and **`cf-ray` / Server** hints.

Where a browser-enabled build is present (`tool-browser` and/or `tool-rendered-fetch`, as with **full-tools**):

- The research **`fetch_research_document`** path (`DocRead` / `DeepRead` / **`auto`**/ **`main`** **`WebFetch`**) can escalate to rendered extraction when bodies look walled or unusually sparse after HTTP success.
- **`WebFetch`** with **`extract`=`raw`**: rejects obvious challenge HTML unless **`rendered_fallback`:** **`true`**; on challenge-like **`4xx`**/ **`429`** when headers/body hint a gate, **`rendered_fallback`** can retry in the browser before failing.
- The **`Browser`** tool supports stealth-oriented launch (**headless UA** cleanup, nodriver-ish args, optional **`MANGOCODE_BROWSER_EXPERT`** shadow hooks), **`navigate`** (runs challenge helpers on load), **`pass_challenge`** (viewport PNG template match loop without re-navigating), and optional **`MANGOCODE_BROWSER_ACCEPT_INSECURE_TLS_UI`** for Chrome’s **`thisisunsafe`** interstitial keystroke (**dev-only** caution).

`WebFetch` stores cache files under `~/.mangocode/web_cache`. Keys include **`extract`**, **`rendered_fallback`**, and related options, with an updated key prefix so stale challenge HTML cached under older builds will not collide.

Environment knobs:

| Variable | Role |
|---|---|
| `MANGOCODE_BROWSER_USER_DATA_DIR` | Optional persistent Chromium profile. |
| `MANGOCODE_BROWSER_LANG` | Passed into stealth launch `Accept-Language` hints. |
| `MANGOCODE_BROWSER_NAV_TIMEOUT_MS` | Extra ms to sleep after navigations settle (bounded). |
| `MANGOCODE_BROWSER_CHALLENGE_NAV_TIMEOUT_MS` | Bounded `wait_for_navigation` wait after checkbox clicks (default 12s, clamped). |
| `MANGOCODE_BROWSER_ACCEPT_INSECURE_TLS_UI` | `1` / `true`: send `thisisunsafe` on Chrome TLS interstitial. |
| `MANGOCODE_BROWSER_EXPERT` | `1` / `true`: expert args plus shadow probing; skips template checkbox matcher. |
| `MANGOCODE_CF_CHALLENGE_MAX_ATTEMPTS` | Max template rounds (≤30, default 6). |
| `MANGOCODE_CF_TEMPLATE_MIN_SCORE` | Minimum correlation for template peaks (default 0.42). |

Rendered browser fallback for `DocRead`/`DeepRead` is optional and uses MangoCode's browser-backed extraction build:

```powershell
cd src-rust
cargo run -p mangocode --features browser
```

Use `--features browser` when you also want the interactive `Browser` tool, or `--features tool-rendered-fetch` when you only want the direct rendered research path. Without either rendered feature, MangoCode still uses HTTP extraction plus native script-data extraction.

The direct `RenderedFetch` tool is only registered by `full-tools` or `tool-rendered-fetch`; the `browser` feature enables rendered fallback and the interactive `Browser` tool without exposing `RenderedFetch` by itself.

If you want to replace MangoCode's native public web search and research tools with an external MCP research/search server, build without the native web/research tool features:

```powershell
cd src-rust
cargo build -p mangocode --no-default-features --features default-tools-no-web-research
```

This removes `WebSearch`, `WebFetch`, `DocSearch`, `DocRead`, `DeepRead`, and `RenderedFetch` from the built-in tool registry. Connected MCP server tools remain available through MangoCode's MCP bridge.

Lifecycle hooks now also support `SessionStart`, `PreCompact`, `PostCompact`, and `SessionEnd` in addition to the existing tool and stop events, so memory and external automations can track full-session behavior instead of only individual tool calls.

## Tool Output Compression

MangoCode compresses noisy command output before it reaches the model while preserving raw logs locally.

- Common reducers cover Rust, Python, JavaScript package managers, TypeScript, `git diff`, `git status`, directory listings, duplicate stack traces, and repeated logs.
- Raw logs are cached under `~/.mangocode/tool-logs`.
- Use a tool's `output_mode: "raw"` when exact full output matters.
- Use `ToolLogRead` to read a cached raw log path referenced in a compressed tool result.

Settings:

```toml
[tool_output]
reduction = "auto" # auto | raw | summary
```

## Mango Memory

Layered memory stores durable facts in project-local SQLite at `.mangocode/layered-memory.sqlite`.

Memory classes:

- `project_fact`
- `user_preference`
- `decision`
- `external_doc`

Privacy controls:

- `<private>...</private>` sections are ignored.
- Common secret shapes are skipped.
- Raw command output is not stored unless summarized.
- `/memory review` shows recent memories before deletion or cleanup.

### Real Local Embeddings

MangoCode uses native FastEmbed by default with `BAAI/bge-base-en-v1.5`, cached under `~/.mangocode/embeddings/fastembed`. The first run may download the model and ONNX Runtime; after that, retrieval runs locally without a cloud embedding API.

Override defaults with:

```powershell
$env:MANGOCODE_EMBEDDINGS_PROVIDER = "fastembed"
$env:MANGOCODE_EMBEDDINGS_MODEL = "BAAI/bge-base-en-v1.5"
```

For custom local embedding engines, set a command fallback:

```powershell
$env:MANGOCODE_EMBEDDINGS_PROVIDER = "command"
$env:MANGOCODE_EMBEDDINGS_COMMAND = "my-local-embedder"
$env:MANGOCODE_EMBEDDINGS_MODEL = "local-neural-embedding"
```

The command receives text on stdin and must print either:

- a JSON array like `[0.12, -0.03, ...]`
- an object with an `embedding` array
- whitespace/comma-separated floats

MangoCode stores these vectors in SQLite and uses dense cosine similarity during retrieval. If FastEmbed or the command provider is unavailable, MangoCode falls back to the always-on local hashed retrieval vector.

## Helpful Settings

```toml
[research]
enable_rendered_fallback = true
prefer_official_sources = true

[attachments]
markdown_extraction_enabled = true
images_raw_by_default = true
ocr_enabled = true
tesseract_lang = "eng"

[tool_output]
reduction = "auto"

[memory]
layered_retrieval = true
embedding_provider = "fastembed"
embedding_model = "BAAI/bge-base-en-v1.5"
```
