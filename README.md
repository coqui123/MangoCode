# MangoCode

MangoCode is a Rust-powered terminal coding agent built for real-world development: interactive sessions, multi-provider model access, MCP tooling, and automation-ready headless runs.

This repo began from the Claurst clean-room effort and has since been heavily modified with substantial runtime, provider, UX, and architecture changes.

<img width="2132" height="1124" alt="image" src="https://github.com/user-attachments/assets/27aec059-6f1c-4427-a727-1d4466c71e47" />

## Highlights

- Provider flexibility without lock-in: use cloud and local models with one CLI (`/connect`, `/model`, `--provider`, `--model`).
- Interactive and headless in one tool: use the TUI for day-to-day coding and the same engine for scripts/CI (`--print`, `--output-format stream-json`).
- MCP-ready by design: connect external tool ecosystems and workflows through built-in MCP support.
- Rust workspace architecture: modular crates for CLI, core runtime, query engine, tools, MCP, and TUI.
- Practical operator workflows: built-in commands for setup, health checks, permissions, tasks, and diagnostics.
- Persistent developer state: settings, auth, and session history live in local config for repeatable workflows.

## Current Status

- Core Rust workspace is active and used as the primary implementation
- Multi-provider support is available
- `/connect` provider onboarding flow is available in-app
- Project is actively evolving

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
```

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

### Structured Streaming Output

```powershell
cd src-rust
cargo run -p mangocode -- --output-format stream-json -p "Generate a concise release note draft for the latest provider refactor."
```

### Useful CI-Safe Flags

- `--print`
- `--output-format stream-json`
- `--max-turns <n>`
- `--provider <id>`
- `--model <id>`
- `--cwd <path>`

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

## Important Notice

This repository does not include proprietary Claude Code TypeScript source code.

MangoCode follows a clean-room style process:

1. Specification phase (`spec/`): behavior, architecture, contracts, and system design are documented.
2. Implementation phase (`src-rust/`): Rust code is implemented from that specification and ongoing independent development.

The guiding principle is idea-expression separation: behavior and interfaces can be reimplemented without copying protected expression.

## Credits

Major credit to the original Claurst effort and research by Kuber Mehta, which provided the initial foundation this repository builds on.

- Claurst repository: https://github.com/kuberwastaken/claude-code
- Original write-up: https://kuber.studio/blog/AI/Claude-Code's-Entire-Source-Code-Got-Leaked-via-a-Sourcemap-in-npm,-Let's-Talk-About-it

MangoCode is now a heavily modified and actively evolving Rust project, and this attribution remains intentionally preserved.

## License

GPL-3.0. See `LICENSE` and `LICENSE.md`.
