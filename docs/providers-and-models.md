# Providers & Models

MangoCode talks to many model backends through one normalized interface in the
[`api`](crates/api.md) crate. This page lists the supported providers, how a model
string is resolved to a provider, and how the model registry works.

---

## How models are resolved

A model can be specified as a bare name (e.g. `sonnet`, `gpt-5`) or as a
fully-qualified `provider/model` string (e.g. `openai/gpt-4o`,
`ollama/llama3.1`). Provider/model parsing only splits on a **known** provider
prefix, so model namespaces like `meta-llama/Llama-3` are not mis-parsed — see
`split_known_model_prefix` in
[`crates/core/src/provider_id.rs`](../src-rust/crates/core/src/provider_id.rs).

`ProviderId` and `ModelId` are branded string newtypes with constants for every
known provider. The default provider when none is specified is `lmstudio`.

Provider selection at runtime:

1. `--provider` / `--model provider/...` if given.
2. `Config.provider` / `Config.model`.
3. The default (`lmstudio`).

Credentials for the chosen provider come from the auth store, the vault, or the
provider's environment variable (see [getting-started.md](getting-started.md#authentication)).

---

## First-class providers

These have dedicated adapters in
[`crates/api/src/providers/`](../src-rust/crates/api/src/providers) implementing the
`LlmProvider` trait (streaming, tool calling, model listing, health check):

| Provider | `ProviderId` | Auth | Notes |
| --- | --- | --- | --- |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Native Messages API; streaming SSE, tool use, extended thinking, vision, prompt caching. |
| Anthropic (OAuth) | `anthropic-max` | OAuth (claude.ai / Console) | Bearer-token auth for Claude Pro/Max. |
| OpenAI | `openai` | `OPENAI_API_KEY` | Chat Completions; auto-routes reasoning models (o-series / gpt-5+) to the Responses API. |
| OpenAI Codex | `openai-codex` | OAuth (ChatGPT) | Codex "Responses" backend; uses an Anthropic⇄Codex adapter. |
| Google Gemini | `google` | `GOOGLE_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` | Streaming SSE, function calling, thinking, vision, video input. |
| Google Vertex | `vertex` | GCP credentials | Vertex AI with OpenAI compatibility. |
| Azure OpenAI | `azure` | `AZURE_API_KEY` + `AZURE_RESOURCE_NAME` | OpenAI-compatible deployments (`api-key` header). |
| AWS Bedrock | `amazon-bedrock` | AWS creds / `AWS_BEARER_TOKEN_BEDROCK` | Claude via the Converse streaming API (SigV4 or bearer). |
| GitHub Copilot | `github-copilot` | `GITHUB_TOKEN` | Chat Completions / Responses API. |
| Cohere | `cohere` | `COHERE_API_KEY` | Command API. |
| MiniMax | `minimax` | `MINIMAX_API_KEY` (+ `MINIMAX_GROUP_ID`) | Tool calling, streaming. |
| Mock | `mock` | — | Canned responses for tests. |

---

## OpenAI-compatible providers

Any OpenAI-compatible endpoint is handled by the generic `OpenAiCompatProvider`
adapter ([`openai_compat.rs`](../src-rust/crates/api/src/providers/openai_compat.rs)),
with named factories in
[`openai_compat_providers.rs`](../src-rust/crates/api/src/providers/openai_compat_providers.rs):

| Provider | `ProviderId` | Env / endpoint |
| --- | --- | --- |
| Ollama | `ollama` | `OLLAMA_HOST` (default `http://localhost:11434`) |
| LM Studio | `lmstudio` | `LM_STUDIO_HOST` (default `http://localhost:1234`) |
| Llama.cpp | `llama-cpp` | `LLAMA_CPP_HOST` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| xAI (Grok) | `xai` | `XAI_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |
| Qwen / DashScope | `qwen` | `QWEN_API_KEY` / `DASHSCOPE_API_KEY` |
| Perplexity | `perplexity` | `PERPLEXITY_API_KEY` |
| Together AI | `together-ai` | `TOGETHER_API_KEY` |
| DeepInfra | `deepinfra` | `DEEPINFRA_API_KEY` |
| Cerebras | `cerebras` | `CEREBRAS_API_KEY` |
| Fireworks | `fireworks` | `FIREWORKS_API_KEY` |
| SambaNova | `sambanova` | `SAMBANOVA_API_KEY` |
| Hugging Face | `huggingface` | `HUGGINGFACE_API_KEY` |
| NVIDIA | `nvidia` | `NVIDIA_API_KEY` |
| Moonshot | `moonshot` | `MOONSHOT_API_KEY` |
| Zhipu (ChatGLM) | `zhipu` | `ZHIPU_API_KEY` |
| Venice | `venice` | `VENICE_API_KEY` |
| SiliconFlow | `siliconflow` | `SILICONFLOW_API_KEY` |
| Nebius | `nebius` | `NEBIUS_API_KEY` |
| Novita | `novita` | `NOVITA_API_KEY` |
| Baseten | `baseten` | `BASETEN_API_KEY` |
| Friendli | `friendli` | `FRIENDLI_API_KEY` |
| Upstage | `upstage` | `UPSTAGE_API_KEY` |
| Stepfun | `stepfun` | `STEPFUN_API_KEY` |
| OVHcloud | `ovhcloud` | `OVHCLOUD_API_KEY` |
| Scaleway | `scaleway` | `SCALEWAY_API_KEY` |
| Vultr | `vultr-ai` | `VULTR_API_KEY` |

> This list reflects the named factories in the source; the generic adapter can
> also point at any other OpenAI-compatible base URL via `--api-base` /
> `provider_configs`.

A special integration, **Copilot-Pirate** (a local M365 Copilot proxy), is wired
through `openai_compat`/`copilot_pirate_tools.rs` and the
`query::copilot_server` auto-start helper.

---

## Model registry

The model registry (`api::model_registry`) describes models with metadata and
pricing in `ModelEntry`:

- `info` — `ModelInfo` (id, provider, context window, max output tokens, reasoning
  levels).
- `cost_input` / `cost_output` / `cost_cache_read` / `cost_cache_write` — USD per
  1M tokens, when known.
- `tool_calling`, `reasoning`, `vision` — capability flags.
- `family`, `status` — e.g. `claude`/`gpt`/`gemini`, and `active`/`beta`/`deprecated`.

It ships with a bundled snapshot of popular models and can persist a refreshed
catalog to disk. A `coding_capability_score()` heuristic (0–100) ranks models for
coding suitability based on tool calling, reasoning, context window, max output,
family, and status — used to surface good defaults in the model picker.

Use `/model` (TUI) or `--list-models` to see what's available for your configured
providers, and `/providers` to manage providers.

---

## Streaming, retries & failover

- **Streaming** — provider adapters parse SSE or JSON-lines into a normalized
  `StreamEvent` stream (`MessageStart`, `TextDelta`, `ThinkingDelta`,
  `InputJsonDelta`, `ContentBlockStop`, `MessageDelta`, `MessageStop`, `Error`).
- **Retries** — `api::retry` retries on 429 (rate limit) and 529 (overloaded),
  honoring `Retry-After` then exponential backoff with jitter (`RetryConfig`).
- **Error classification** — `api::error_handling` detects context-overflow across
  providers and maps HTTP errors to a structured `ProviderError` with actionable
  `ProviderDiagnostic` suggestions.
- **Failover** — `stream_with_failover` tries providers in priority order, skipping
  non-retryable errors (auth, model-not-found, quota). The query loop can also
  switch to `fallback_model` on overload.

Full details in [crates/api.md](crates/api.md).
