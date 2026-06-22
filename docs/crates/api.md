# Crate: `api`

Path: [`src-rust/crates/api`](../../src-rust/crates/api) ·
Core: [`src/lib.rs`](../../src-rust/crates/api/src/lib.rs)

`api` is the model-provider layer. It defines a normalized request/response/stream
model, adapts 40+ providers to it, parses streaming responses, transforms between
MangoCode's format and provider wire formats, manages a model registry, and handles
retries, error classification, and failover. It also contains the Anthropic-native
client used directly by the query loop.

See [providers-and-models.md](../providers-and-models.md) for the user-facing
provider list.

---

## Provider abstraction

The central trait is `LlmProvider` ([`provider.rs`](../../src-rust/crates/api/src/provider.rs)):

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &ProviderId;
    fn name(&self) -> &str;
    async fn create_message(&self, request: ProviderRequest) -> Result<ProviderResponse, ProviderError>;
    async fn create_message_stream(&self, request: ProviderRequest)
        -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError>;
    async fn health_check(&self) -> Result<ProviderStatus, ProviderError>;
    fn capabilities(&self) -> ProviderCapabilities;
}
```

### Normalized types (`provider_types.rs`)

- **`ProviderRequest`** — model, messages, system prompt, tools, sampling params
  (`temperature`/`top_p`/`top_k`/`stop_sequences`), `thinking` config, and
  free-form `provider_options`.
- **`ProviderResponse`** — id, `Vec<ContentBlock>`, `StopReason`, `UsageInfo`, model.
- **`StreamEvent`** — `MessageStart`, `ContentBlockStart`, `TextDelta`,
  `ThinkingDelta`, `InputJsonDelta`, `SignatureDelta`, `ContentBlockStop`,
  `MessageDelta`, `MessageStop`, `ReasoningDelta`, `Error`.
- **`StopReason`** — `EndTurn`, `StopSequence`, `MaxTokens`, `ToolUse`,
  `ContentFiltered`, `Other`.
- **`ProviderCapabilities`** — feature flags (streaming, tools, thinking, image/pdf/
  audio/video input, caching, structured output) and `system_prompt_style`
  (`TopLevel` / `SystemMessage` / `SystemInstruction`).
- **`AuthMethod`** — `ApiKey`, `Bearer`, `AwsCredentials`, `OAuth`, `None`.

### Auth (`auth.rs`)

`AuthProvider` handles credential retrieval, login, and logout; `LoginFlow`
describes interactive steps (OAuth url / API key / none).

---

## Providers

Concrete adapters live under
[`providers/`](../../src-rust/crates/api/src/providers): `anthropic`,
`anthropic_oauth`, `openai`, `openai_codex`, `google`, `azure`, `bedrock`,
`cohere`, `copilot`, `minimax`, `vertex_openai`, `mock`, plus the generic
`openai_compat` adapter and its named factories in `openai_compat_providers.rs`.
`message_normalization.rs` and `request_options.rs` provide shared normalization.

The `ProviderRegistry` ([`registry.rs`](../../src-rust/crates/api/src/registry.rs))
holds `Arc<dyn LlmProvider>` per provider with a default, plus convenience builders
that register a provider only when its credentials are present
(`with_openai_if_key_set`, `with_azure_if_configured`, …).

---

## Streaming & transformation

- **`stream_parser.rs`** — `StreamParser` trait with SSE and JSON-lines
  implementations that turn an HTTP response body into a `StreamEvent` stream.
- **`transform.rs` + `transformers/`** — `MessageTransformer` converts a
  `ProviderRequest` to a provider's wire JSON and back. Implementations:
  `transformers/anthropic.rs` (Messages API + cache control) and
  `transformers/openai_chat.rs` (Chat Completions and the Responses API for
  reasoning models).
- **`codex_adapter.rs`** — translates between the Anthropic Messages format and the
  OpenAI Codex "Responses" API (request building, SSE consolidation, response
  blocks).

---

## Anthropic-native client

`AnthropicClient` (in `lib.rs`) is the client the query loop uses directly:

- `create_message(...)` — non-streaming `/v1/messages`.
- `create_message_stream(...)` — streaming; parses SSE frames into
  `AnthropicStreamEvent`s (`MessageStart`, `ContentBlockStart/Delta/Stop`,
  `MessageDelta`, `MessageStop`, `Error`, `Ping`).
- **`StreamAccumulator`** — collects stream events into a complete `Message` +
  `UsageInfo` + stop reason, repairing truncated tool-call JSON along the way.

It supports API-key and bearer (OAuth) auth, prompt caching split at the system
prompt boundary, retry on overload, a per-process workload billing tag, and
**CCH** request signing (`cch.rs`) — an xxHash-based billing header used to verify
legitimate clients and gate features like fast mode.

---

## Model registry

`model_registry.rs` describes models with metadata + pricing (`ModelEntry`) and
ranks them for coding suitability (`coding_capability_score`). It ships a bundled
snapshot and can persist a refreshed catalog. See
[providers-and-models.md](../providers-and-models.md#model-registry).

---

## Retries, errors & failover

- **`retry.rs`** — `retry_request` retries on 429/529, honoring `Retry-After` then
  exponential backoff with jitter (`RetryConfig`: defaults around 8 retries, 2s
  initial, 300s cap).
- **`error_handling.rs`** — `is_context_overflow` detects context-window errors
  across providers; `parse_error_response` maps HTTP errors to `ProviderError`;
  `stream_with_failover` tries providers in order, skipping non-retryable failures.
- **`provider_error.rs`** — structured `ProviderError` variants (context overflow,
  rate limited, auth failed, quota exceeded, model not found, server error, invalid
  request, content filtered, stream error) and a user-facing `ProviderDiagnostic`
  with recovery `suggestions`.
