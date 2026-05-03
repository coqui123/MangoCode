# MangoCode

MangoCode is the open, multi-provider coding agent for serious local and hybrid workflows.

It is a Rust-powered terminal coding agent built for real-world development: interactive pair-programming, automation-ready headless runs, MCP-connected tooling, local and cloud model routing, and durable project memory.

If you want one agent that can work in the terminal, plug into external tool ecosystems, run against multiple model providers, and stay useful in CI or remote environments, MangoCode is built for that job.

<img width="2132" height="1124" alt="image" src="https://github.com/user-attachments/assets/27aec059-6f1c-4427-a727-1d4466c71e47" />

## Why MangoCode

- Open and model-flexible: route work across Anthropic, OpenAI, Google, Qwen, Ollama, Vertex, OpenAI-compatible backends, and more without locking your workflow to a single vendor.
- Built for local and hybrid development: use the TUI for day-to-day coding, run headless in CI, connect MCP servers, and keep the same core runtime across both.
- Strong native tooling story: file editing, shell execution, research tools, MCP resources, skills, tasks, worktrees, and review flows live in one system instead of being bolted on as separate apps.
- Serious context handling: Smart Attachments, OCR fallbacks, Mango Research, layered memory, and tool-output compression help the agent stay useful in large real repositories.
- Automation-ready: structured `stream-json` output, background tasks, diagnostics, permissions, and provider controls make MangoCode practical for repeatable workflows.
- Rust workspace architecture: modular crates for CLI, core runtime, query engine, providers, tools, MCP, plugins, and TUI make the system extensible instead of monolithic.

## Kick-Ass Features

- Multi-provider routing without lock-in: switch models with `/connect`, `/model`, `--provider`, and `--model`.
- Interactive and headless in one tool: use the terminal UI locally, then reuse the same engine in scripts and CI with `--print` and `--output-format stream-json`.
- MCP-ready by design: connect local `stdio` servers and hosted remote MCP servers at the same time.
- Native intelligence layer: Smart Attachments, Mango Research, output compression, layered memory, and local embeddings are documented in [docs/mango-intelligence.md](docs/mango-intelligence.md).
- Practical operator workflows: built-in commands for setup, health checks, permissions, tasks, review, diagnostics, and provider management.
- Persistent developer state: settings, auth, sessions, cached docs, attachment extracts, and tool logs live in local MangoCode data directories.
- Local-first options when you want them: Ollama, local OCR, local embeddings, vault-backed credentials, and headless-friendly provider selection.

## Current Status

- Core Rust workspace is active and used as the primary implementation.
- Multi-provider support is available today.
- `/connect` provider onboarding is available in-app.
- Headless automation and structured output are available today.
- MCP support, memory, research, and attachment processing are already integrated.
- Project is actively evolving and still being sharpened.

## Installation and Running

### Prerequisites

- Rust stable toolchain
- Cargo
- At least one provider credential, or a local model backend (for example Ollama)

### Run From Source

```powershell
cd src-rust
cargo run -p mangocode
```

### Optional: Install Binary

```powershell
cd src-rust
cargo install --path crates/cli
```

## Fast Onboarding

1. Start MangoCode:

```powershell
cd src-rust
cargo run -p mangocode
```

If installed via Cargo, run `mangocode` directly.

2. Inside MangoCode, open provider onboarding:

```text
/connect
```

3. Pick provider and model:

```text
/model
```

4. Verify runtime health:

```text
/status
/doctor
```

## Provider Setup

MangoCode supports provider selection through:

- `/connect` (recommended interactive flow)
- environment variables
- CLI flags (for example `--provider` and `--model`)

### Common Environment Variables

```powershell
# Anthropic
$env:ANTHROPIC_API_KEY = "your_key"

# OpenAI
$env:OPENAI_API_KEY = "your_key"

# Google (Gemini API)
$env:GOOGLE_API_KEY = "your_key"
```

### OpenAI Codex (OAuth) vs OpenAI API key

MangoCode treats these as **separate** auth paths:

- **OpenAI Codex (OAuth)** (`openai-codex`, alias `codex`): sign in with your **ChatGPT** account so Codex requests use your ChatGPT plan. Run `/connect`, choose **OpenAI Codex (OAuth)**, complete the browser login (PKCE + `http://localhost:1455/auth/callback`). Tokens are stored under `~/.mangocode/auth.json` (and mirrored to `~/.mangocode/codex_tokens.json`). The app refreshes the access token automatically before expiry when a refresh token is present. The query engine uses a **synthesized** stream over one Codex HTTP response (no live SSE from the Codex endpoint yet).
- **OpenAI** (`openai`): classic **API key** from [platform.openai.com](https://platform.openai.com/api-keys) — usage-based API billing. Use this for automation/CI or when you want platform billing instead of ChatGPT subscription access.

**Remote / headless:** browser + localhost OAuth often fails over SSH, GitHub Codespaces, or some Dev Containers. MangoCode detects common cases and shows a targeted message; use API-key OpenAI on those hosts, or authenticate on a local machine first.

**Optional:** to migrate tokens from the official Codex CLI, you can use the library helper `mangocode_core::oauth_config::import_codex_cli_auth_json` on a JSON file you export explicitly (MangoCode does not read `~/.codex/auth.json` automatically).

### Quick Provider Runs

```powershell
cd src-rust

# Anthropic default model
cargo run -p mangocode

# OpenAI
cargo run -p mangocode -- --provider openai --model gpt-4o

# Google
cargo run -p mangocode -- --provider google --model gemini-2.5-pro

# Local Ollama
cargo run -p mangocode -- --provider ollama --model ollama/llama3.2

# Local Ollama — Gemma 4 (multimodal, native function-calling)
ollama pull gemma4:e4b   # small/local-friendly (~128K ctx)
cargo run -p mangocode -- --provider ollama --model ollama/gemma4:e4b
```

### Ollama Notes And Troubleshooting

MangoCode talks to Ollama via its OpenAI-compatible `/v1/chat/completions`
endpoint. This works with both vanilla and "thinking" models, but a few
behaviours are worth knowing.

**Inline `<think>` reasoning.** Local thinking models (Qwen3 / DeepSeek-R1
distills / `gpt-oss:20b` / Modelfiles tagged `Thinking`) typically stream
their chain-of-thought wrapped inline as `<think>...</think>` inside
`delta.content` rather than in a separate `reasoning_content` field. MangoCode
strips those wrappers automatically: the inner text is forwarded as
reasoning/progress events so the TUI no longer freezes on a "Calling model..."
spinner while the model is thinking, and the final visible answer (the text
after `</think>`) is emitted as normal assistant content.

**Tool calling.** Not every Ollama model handles native OpenAI-style `tools`
arrays. MangoCode keeps a curated allowlist of known-good families
(`llama3.x`, `llama4`, `qwen2.5`, `qwen3`, `mistral-nemo`, `mistral-large`,
`mixtral`, `command-r`, `firefunction`, `hermes3`, `granite`, `smollm2`,
`gpt-oss`, `gemma4`) and silently drops the `tools` array for any other
model so the request does not stall. If you have a custom Modelfile that
genuinely supports tools, set `MANGOCODE_OLLAMA_FORCE_TOOLS=1` to bypass
the gate.

**Autostart.** When `OLLAMA_HOST` is unset and port `11434` is free,
MangoCode runs `ollama serve` to start a local daemon. The listen address
is governed entirely by `OLLAMA_HOST` (or the daemon default
`127.0.0.1:11434`); MangoCode does **not** pass `--port` to `ollama serve`
because no released `ollama` build accepts that flag.

**Dynamic model discovery.** When you open the `/model` picker with the
Ollama provider selected, MangoCode queries the local daemon at
`GET /api/tags` (the same endpoint backing `ollama list`) and shows the
models that are actually installed on your machine, including parameter
size, quantisation, and on-disk size when the daemon reports them. Each
discovered tag is exposed as `ollama/<tag>` (e.g. `ollama/qwen3:8b`,
`ollama/llama3.2:latest`). The lookup uses a 3-second timeout — if the
daemon is unreachable the picker falls back to a static suggestion list
prefixed with a "start `ollama serve`" diagnostic, and if the daemon is
running but has no models pulled the picker shows a hint to run
`ollama pull <model>`. Verify what the picker should see with:

```bash
curl http://127.0.0.1:11434/api/tags
```

**Quick connectivity checks.** If the TUI hangs on a model request, first
verify the daemon directly:

```bash
# Basic reachability
curl http://127.0.0.1:11434/api/tags

# OpenAI-compatible streaming smoke test
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:8b","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

If either request succeeds but MangoCode still hangs, run with the debug
flag to see how each chunk is classified:

```bash
MANGOCODE_DEBUG_OLLAMA_STREAM=1 cargo run -p mangocode -- --provider ollama --model qwen3:8b
```

**`OLLAMA_HOST` accepts** bare `host:port` (`127.0.0.1:11434`), full URLs
(`http://127.0.0.1:11434`), or pre-suffixed (`http://127.0.0.1:11434/v1`) —
they all normalise to the same base URL.

**Stalled-stream timeout.** If no data arrives from the model for 120 s
MangoCode aborts the request with a clear error rather than hanging
indefinitely. Override with `MANGOCODE_OLLAMA_IDLE_TIMEOUT_MS=300000` for
slow first-token times on under-resourced machines, or pull a Modelfile
with a larger `num_ctx` if the model is paging.

**Gemma 4 setup.** Gemma 4 is a multimodal (text + image) family with
native OpenAI-style function calling, so it is on the tool-capable
allowlist out of the box. Pull whichever variant fits your hardware:

```bash
# Default tag — picks the recommended size for your machine
ollama run gemma4

# Small / local-friendly (~128K context)
ollama run gemma4:e2b
ollama run gemma4:e4b

# Workstation (~256K context)
ollama run gemma4:26b
ollama run gemma4:31b
ollama run gemma4:31b-cloud

# Confirm it appears in /api/tags so the /model picker can discover it
curl http://127.0.0.1:11434/api/tags
```

Then point MangoCode at the tag you want:

```bash
cargo run -p mangocode -- --provider ollama --model ollama/gemma4
cargo run -p mangocode -- --provider ollama --model ollama/gemma4:e4b
cargo run -p mangocode -- --provider ollama --model ollama/gemma4:31b
```

Gemma 4's reasoning is emitted through OpenAI-Harmony channel tokens
(`<|channel>thought ... <channel|>`), not the Qwen-style
`<think>...</think>` wrapper that MangoCode strips automatically. Effort /
extended-thinking controls therefore have no effect on `gemma4` tags
today — the model still answers, just without surfacing its hidden
chain-of-thought as MangoCode "reasoning" events. To force-enable
thinking on a custom Modelfile, prefix the system prompt with `<|think|>`
yourself.

### Google Vertex (ADC) Auth Flow

Use this flow for Vertex-hosted Gemini models via Application Default Credentials.

```powershell
# 1) Sign in
gcloud auth login

# 2) Initialize/update ADC credentials
gcloud auth application-default login

# 3) Vertex provider environment
$env:VERTEX_PROJECT_ID = "your-gcp-project-id"
$env:VERTEX_LOCATION = "us-central1"
$env:VERTEX_MODEL = "google/gemini-2.5-pro"
$env:VERTEX_AUTH_MODE = "adc"

# 4) Run
cd src-rust
cargo run -p mangocode -- --provider google-vertex --model google/gemini-2.5-pro
```

Notes:

- `VERTEX_PROJECT_ID` is the required key variable.
- `VERTEX_LOCATION` is optional (default: `us-central1`).
- `VERTEX_MODEL` is optional (default: `google/gemini-2.5-flash`).
- `VERTEX_AUTH_MODE` is optional (default: `adc`).
- For multi-project setups, only switch `VERTEX_PROJECT_ID` per project.

## Headless And Automation

### Single Prompt (Non-Interactive)

```powershell
cd src-rust
cargo run -p mangocode -- -p "Summarize the crate structure in 6 bullets."
```

### Using Vault Credentials In Headless Mode

If you store provider API keys in the local encrypted vault, headless runs (for example `-p` / `--print`) can unlock the vault using one of these flags:

- `--vault-prompt`: prompt for the vault passphrase (even in headless mode).
- `--vault-passphrase <PASSPHRASE>`: unlock non-interactively.

Security note: `--vault-passphrase` may leak via shell history and process lists. Prefer `--vault-prompt` when possible.

### Structured Streaming Output

```powershell
cd src-rust
cargo run -p mangocode -- --output-format stream-json -p "Generate a concise release note draft for the latest provider refactor."
```

### Saving `stream-json` Output (Clean JSONL)

If you use `cargo run`, Cargo will print build and `Running ...` lines which will pollute a `.jsonl` file. For clean JSONL logs, build once and run the binary directly:

```powershell
cd src-rust
cargo build -p mangocode
.\target\debug\mangocode.exe --output-format stream-json -p "Hello from headless stream-json." |& Tee-Object -FilePath .\mangocode-stream.jsonl
```

PowerShell note: use `|&` (or `2>&1 |`) to capture both stdout and stderr into the log.

### Debug: Dump Raw Provider SSE Frames (OpenAI-Compatible Providers)

When debugging tool calling, it's often necessary to see the raw streaming frames (`data: ...`) coming back from the provider (especially for OpenAI-compatible backends that differ slightly from OpenAI's tool-call streaming).

Set `MANGOCODE_DUMP_OPENAI_COMPAT_SSE=1` and enable trace logs for the wire target:

```powershell
cd src-rust
$env:MANGOCODE_DUMP_OPENAI_COMPAT_SSE = "1"
$env:RUST_LOG = "mangocode_api::providers::openai_compat::wire=trace"
.\target\debug\mangocode.exe --output-format stream-json -p "Call a tool, then stop." |& Tee-Object -FilePath .\provider-wire.log
```

Notes:

- This logs to stderr, so it won’t corrupt `--output-format stream-json` stdout.
- The dump is truncated per frame and may include sensitive content—use with care.

Example: Qwen (DashScope) headless stream output using vault unlock:

```powershell
cd src-rust
.\target\debug\mangocode.exe --provider qwen --model qwen3.6-plus-2026-04-02 --output-format stream-json --vault-prompt --max-turns 2 -p "Call one tool (Grep for 'openai_compat' in the repo), then stop." |& Tee-Object -FilePath .\qwen-stream.jsonl
```

### Useful CI-Safe Flags

- `--print`
- `--output-format stream-json`
- `--max-turns <n>`
- `--provider <id>`
- `--model <id>`
- `--cwd <path>`
- `--vault-prompt`
- `--vault-passphrase <PASSPHRASE>`
- `--qwen-preserve-thinking`

## Useful In-App Commands

- `/help` - command discovery and usage
- `/connect` - provider setup flow
- `/model` - model selection
- `/mcp` - MCP server status and actions
- `/permissions` - permission mode and policy details
- `/tasks` - background task status
- `/status` - runtime/session status
- `/doctor` - diagnostics and setup checks
- `/review` - review-oriented workflow

## Configuration And Data Locations

MangoCode stores user-level data under the `~/.mangocode` directory.

- `~/.mangocode/settings.json` - global settings
- `~/.mangocode/auth.json` - stored provider credentials/tokens
- `~/.mangocode/sessions.db` - session persistence

Project-level overrides are supported via:

- `.mangocode/settings.json`
- `.mangocode/settings.jsonc`

## MCP Configuration

MangoCode can connect to both local `stdio` MCP servers and hosted remote MCP servers at the same time.

Example `~/.mangocode/settings.json`:

```json
{
  "config": {
    "mcp_servers": [
      {
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "type": "stdio"
      },
      {
        "name": "pipedream",
        "url": "https://remote.mcp.pipedream.net/v3",
        "type": "pipedream",
        "pipedream": {
          "client_id": "${PIPEDREAM_CLIENT_ID}",
          "client_secret": "${PIPEDREAM_CLIENT_SECRET}",
          "project_id": "${PIPEDREAM_PROJECT_ID}",
          "environment": "${PIPEDREAM_ENVIRONMENT:-development}",
          "external_user_id": "${PIPEDREAM_EXTERNAL_USER_ID:-local-dev}",
          "app_slug": "${PIPEDREAM_APP_SLUG:-github}",
          "tool_mode": "${PIPEDREAM_TOOL_MODE:-sub-agent}"
        }
      }
    ]
  }
}
```

Notes:

- Use `env` for `stdio` server environment variables.
- Use `headers` for remote HTTP/SSE MCP servers such as Pipedream.
- Use `type: "pipedream"` when you want MangoCode to mint and refresh the Pipedream access token from client credentials automatically.
- `${VAR}` and `${VAR:-default}` expansion works in `command`, `args`, `env`, `url`, `headers`, and the `pipedream` block.
- Optional Pipedream fields include `tool_mode`, `conversation_id`, `account_id`, `scope`, `app_discovery`, and `token_url`.
- Per-server MCP config overrides global defaults. Otherwise MangoCode prefers the encrypted vault, then environment variables, then `~/.mangocode/pipedream.json`.
- Recommended vault keys are `pipedream-client-id`, `pipedream-client-secret`, `pipedream-project-id`, `pipedream-environment`, `pipedream-external-user-id`, `pipedream-app-slug`, `pipedream-app-discovery`, `pipedream-tool-mode`, `pipedream-conversation-id`, `pipedream-account-id`, `pipedream-scope`, `pipedream-mcp-url`, and `pipedream-token-url`.
- `/pipedream setup` stores the collected Pipedream values in the encrypted MangoCode vault and saves non-secret fallback defaults to `~/.mangocode/pipedream.json`.
- The Pipedream CLI (`pd`) is optional. MangoCode does not require it, invoke it, or auto-install it at runtime.
- If you use `pd init connect`, treat it as a credential/bootstrap helper only. Copy the resulting values into your shell environment or MangoCode settings because MangoCode does not automatically load another project's `.env` file.
- Hosted Pipedream MCP is a good fit as a second MCP backend for SaaS integrations while keeping your existing local/custom MCP servers in place. If no `app_slug` is set, MangoCode enables Pipedream app discovery by default.

## Repository Layout

- `src-rust/` - Rust workspace root
- `src-rust/crates/cli/` - CLI entry point and runtime orchestration
- `src-rust/crates/core/` - shared types, config, permissions, context, history
- `src-rust/crates/api/` - provider clients/adapters and model/provider registry
- `src-rust/crates/tools/` - tool implementations
- `src-rust/crates/query/` - query loop and agent orchestration
- `src-rust/crates/tui/` - terminal UI and overlays
- `src-rust/crates/commands/` - slash command implementations
- `src-rust/crates/mcp/` - MCP runtime and connectors
- `spec/` - behavior and architecture references

## Development Workflow

```powershell
cd src-rust

# Build
cargo build

# Run tests
cargo test

# Run the app
cargo run -p mangocode
```

## Project Background And Clean-Room Notice

This repository does not include proprietary Claude Code TypeScript source code.

MangoCode follows a clean-room style process:

1. Specification phase (`spec/`): behavior, architecture, contracts, and system design are documented.
2. Implementation phase (`src-rust/`): Rust code is implemented from that specification and ongoing independent development.

The guiding principle is idea-expression separation: behavior and interfaces can be reimplemented without copying protected expression.

## Credits And Lineage

Major credit to the original Claurst effort and research by Kuber Mehta, which provided the initial foundation this repository builds on.

- Claurst repository: https://github.com/kuberwastaken/claude-code
- Original write-up: https://kuber.studio/blog/AI/Claude-Code's-Entire-Source-Code-Got-Leaked-via-a-Sourcemap-in-npm,-Let's-Talk-About-it

MangoCode is now a heavily modified and actively evolving Rust project, and this attribution remains intentionally preserved.

## License

GPL-3.0. See `LICENSE` and `LICENSE.md`.
