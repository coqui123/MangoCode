// providers/openai_compat_providers.rs — Factory functions for all
// OpenAI-compatible provider instances.
//
// Each function constructs a pre-configured [`OpenAiCompatProvider`] for a
// specific service.  API keys are read from environment variables; if the
// variable is absent or empty the provider is still constructed but
// `health_check()` will return `ProviderStatus::Unavailable`.

use std::fs;
use std::path::PathBuf;

use dirs;
use mangocode_core::provider_id::ProviderId;
use url::Url;

use super::openai_compat::{OpenAiCompatProvider, ProviderQuirks};

// ---------------------------------------------------------------------------
// Local / self-hosted providers (no API key required)
// ---------------------------------------------------------------------------

/// Ollama — local inference server.
/// Reads `OLLAMA_HOST` for the base URL; defaults to `http://localhost:11434`.
pub fn ollama() -> OpenAiCompatProvider {
    let host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let base_url = normalize_ollama_base_url(&host);
    OpenAiCompatProvider::new(ProviderId::OLLAMA, "Ollama", base_url).with_quirks(ProviderQuirks {
        overflow_patterns: vec![
            "prompt too long".to_string(),
            "exceeded.*context length".to_string(),
        ],
        // Ollama / Qwen-thinking models stream reasoning inline as `<think>...</think>`
        // inside `delta.content`. The streaming parser strips the wrapper and
        // forwards the inner text as ReasoningDelta events.
        inline_think_tags: true,
        ..Default::default()
    })
}

/// Normalize a user-supplied `OLLAMA_HOST` value into a full
/// `http(s)://host:port/v1` base URL.
///
/// Accepts:
///   - Bare host:port              (e.g. `127.0.0.1:11434`)
///   - With scheme                 (e.g. `http://127.0.0.1:11434`)
///   - With trailing slash         (e.g. `http://127.0.0.1:11434/`)
///   - Already-suffixed `/v1`      (e.g. `http://127.0.0.1:11434/v1`)
///   - Already-suffixed `/v1/`     (e.g. `http://127.0.0.1:11434/v1/`)
///
/// Falls through to `format!("{host}/v1")` for exotic paths so reverse-proxy
/// users can still mount Ollama under a custom prefix.
pub(crate) fn normalize_ollama_base_url(host: &str) -> String {
    let trimmed = host.trim();
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{}", trimmed)
    };
    let no_trail = with_scheme.trim_end_matches('/').to_string();
    if no_trail.ends_with("/v1") {
        no_trail
    } else {
        format!("{}/v1", no_trail)
    }
}

/// Derive the native Ollama API base URL (without `/v1`) from a user-supplied
/// `OLLAMA_HOST` value.
///
/// The OpenAI-compatible endpoint lives at `<host>/v1/...`, while native
/// Ollama endpoints (`/api/tags`, `/api/show`, etc.) live at `<host>/...`.
/// This is used by [`discover_installed_ollama_models`] to query
/// `GET /api/tags`.
pub fn native_ollama_base_url(host: &str) -> String {
    let trimmed = host.trim();
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{}", trimmed)
    };
    let no_trail = with_scheme.trim_end_matches('/').to_string();
    // Strip a trailing `/v1` (and any duplicate trailing slash) so we land on
    // the native API base. `trim_end_matches` handles `/v1` only; we already
    // stripped the trailing slash above.
    if let Some(stripped) = no_trail.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        no_trail
    }
}

/// Read the configured `OLLAMA_HOST` value (or the default) and return the
/// native API base URL.
pub fn ollama_native_base_from_env() -> String {
    let host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    native_ollama_base_url(&host)
}

fn is_local_ollama_host(host: &str) -> bool {
    let trimmed = host.trim();
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{}", trimmed)
    };

    Url::parse(&with_scheme)
        .ok()
        .and_then(|url| url.host_str().map(|h| h == "127.0.0.1" || h == "localhost"))
        .unwrap_or(false)
}

fn ollama_model_store_dir() -> Option<PathBuf> {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    if !is_local_ollama_host(&host) {
        return None;
    }

    let mut path = dirs::home_dir()?;
    path.push(".ollama");
    path.push("models");
    Some(path)
}

fn discover_installed_ollama_models_from_local_store() -> Vec<OllamaInstalledModel> {
    let models_dir = match ollama_model_store_dir() {
        Some(dir) => dir,
        None => return Vec::new(),
    };

    let mut installed = Vec::new();
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if !file_type.is_dir() && !file_type.is_symlink() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.is_empty() {
                continue;
            }
            installed.push(OllamaInstalledModel {
                name,
                modified_at: None,
                size: None,
                digest: None,
                details: None,
            });
        }
    }
    installed
}

/// A single model installed on the local Ollama server, as returned by
/// `GET /api/tags`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OllamaInstalledModel {
    /// Tag-qualified model name, e.g. `qwen3:8b` or `llama3.2:latest`.
    pub name: String,
    #[serde(default)]
    pub modified_at: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaInstalledModel>,
    #[serde(default)]
    tags: Vec<OllamaInstalledModel>,
}

/// Parse a JSON body returned by `GET /api/tags` into a list of installed
/// models. Supports the three common shapes observed from Ollama servers:
/// 1) object with `models` array, 2) object with `tags` array, and
/// 3) array of model objects.
///
/// Returns an empty list when the body is not valid JSON or contains no
/// recognized model array.
pub fn parse_ollama_tags_response(body: &str) -> Vec<OllamaInstalledModel> {
    serde_json::from_str::<OllamaTagsResponse>(body)
        .ok()
        .and_then(|r| {
            if !r.models.is_empty() {
                Some(r.models)
            } else if !r.tags.is_empty() {
                Some(r.tags)
            } else {
                Some(Vec::new())
            }
        })
        .or_else(|| serde_json::from_str::<Vec<OllamaInstalledModel>>(body).ok())
        .unwrap_or_default()
}

/// Errors returned when discovering installed Ollama models.
#[derive(Debug, thiserror::Error)]
pub enum OllamaDiscoveryError {
    #[error("Ollama server not reachable at {base_url}: {source}")]
    Unreachable {
        base_url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("Ollama /api/tags returned HTTP {status}")]
    HttpStatus { status: u16 },
    #[error("Failed to read Ollama response body: {0}")]
    Body(#[source] reqwest::Error),
}

/// Query `GET <base>/api/tags` against the local Ollama server and return the
/// list of installed models.
///
/// Uses a short total timeout (default 3 seconds) so a hung server cannot
/// block the model picker. On any error the caller is expected to fall back
/// to a static list and surface a diagnostic to the user.
pub async fn discover_installed_ollama_models(
    base_url: &str,
    timeout: std::time::Duration,
) -> Result<Vec<OllamaInstalledModel>, OllamaDiscoveryError> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| OllamaDiscoveryError::Unreachable {
            base_url: base_url.to_string(),
            source: e,
        })?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| OllamaDiscoveryError::Unreachable {
            base_url: base_url.to_string(),
            source: e,
        })?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(OllamaDiscoveryError::HttpStatus { status });
    }
    let body = resp.text().await.map_err(OllamaDiscoveryError::Body)?;
    let installed = parse_ollama_tags_response(&body);
    if !installed.is_empty() {
        return Ok(installed);
    }

    let local = discover_installed_ollama_models_from_local_store();
    if !local.is_empty() {
        return Ok(local);
    }

    Ok(installed)
}

/// LM Studio — local OpenAI-compatible server.
/// Reads `LM_STUDIO_HOST` for the base URL; defaults to `http://localhost:1234`.
pub fn lm_studio() -> OpenAiCompatProvider {
    let host =
        std::env::var("LM_STUDIO_HOST").unwrap_or_else(|_| "http://localhost:1234".to_string());
    let base_url = format!("{}/v1", host.trim_end_matches('/'));
    OpenAiCompatProvider::new(ProviderId::LM_STUDIO, "LM Studio", base_url).with_quirks(
        ProviderQuirks {
            overflow_patterns: vec!["greater than the context length".to_string()],
            ..Default::default()
        },
    )
}

/// llama.cpp — lightweight C++ inference server.
/// Reads `LLAMA_CPP_HOST` for the base URL; defaults to `http://localhost:8080`.
pub fn llama_cpp() -> OpenAiCompatProvider {
    let host =
        std::env::var("LLAMA_CPP_HOST").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let base_url = format!("{}/v1", host.trim_end_matches('/'));
    OpenAiCompatProvider::new(ProviderId::LLAMA_CPP, "llama.cpp", base_url).with_quirks(
        ProviderQuirks {
            overflow_patterns: vec!["exceeds the available context size".to_string()],
            ..Default::default()
        },
    )
}

// ---------------------------------------------------------------------------
// Remote / cloud providers (API key required)
// ---------------------------------------------------------------------------

/// DeepSeek — supports reasoning output via `reasoning_content` field.
/// Reads `DEEPSEEK_API_KEY`.
pub fn deepseek() -> OpenAiCompatProvider {
    let key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::DEEPSEEK,
        "DeepSeek",
        "https://api.deepseek.com/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        reasoning_field: Some("reasoning_content".to_string()),
        overflow_patterns: vec!["maximum context length is".to_string()],
        include_usage_in_stream: true,
        max_tokens_cap: Some(8192),
        ..Default::default()
    })
}

/// Groq — fast inference cloud.  Reads `GROQ_API_KEY`.
pub fn groq() -> OpenAiCompatProvider {
    let key = std::env::var("GROQ_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(ProviderId::GROQ, "Groq", "https://api.groq.com/openai/v1")
        .with_api_key(key)
        .with_quirks(ProviderQuirks {
            overflow_patterns: vec!["reduce the length of the messages".to_string()],
            include_usage_in_stream: true,
            ..Default::default()
        })
}

/// xAI (Grok).  Reads `XAI_API_KEY`.
pub fn xai() -> OpenAiCompatProvider {
    let key = std::env::var("XAI_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(ProviderId::XAI, "xAI (Grok)", "https://api.x.ai/v1")
        .with_api_key(key)
        .with_quirks(ProviderQuirks {
            overflow_patterns: vec!["maximum prompt length is".to_string()],
            ..Default::default()
        })
}

/// DeepInfra — hosted open-weight models.  Reads `DEEPINFRA_API_KEY`.
pub fn deepinfra() -> OpenAiCompatProvider {
    let key = std::env::var("DEEPINFRA_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::DEEPINFRA,
        "DeepInfra",
        "https://api.deepinfra.com/v1/openai",
    )
    .with_api_key(key)
}

/// Cerebras — wafer-scale inference.  Reads `CEREBRAS_API_KEY`.
pub fn cerebras() -> OpenAiCompatProvider {
    let key = std::env::var("CEREBRAS_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::CEREBRAS,
        "Cerebras",
        "https://api.cerebras.ai/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        include_usage_in_stream: true,
        ..Default::default()
    })
}

/// Together AI — hosted open-source models.  Reads `TOGETHER_API_KEY`.
pub fn together_ai() -> OpenAiCompatProvider {
    let key = std::env::var("TOGETHER_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::TOGETHER_AI,
        "Together AI",
        "https://api.together.xyz/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        include_usage_in_stream: true,
        ..Default::default()
    })
}

/// Perplexity — search-augmented LLM API.  Reads `PERPLEXITY_API_KEY`.
pub fn perplexity() -> OpenAiCompatProvider {
    let key = std::env::var("PERPLEXITY_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::PERPLEXITY,
        "Perplexity",
        "https://api.perplexity.ai",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        include_usage_in_stream: true,
        ..Default::default()
    })
}

/// Venice AI — privacy-focused inference.  Reads `VENICE_API_KEY`.
pub fn venice() -> OpenAiCompatProvider {
    let key = std::env::var("VENICE_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::VENICE,
        "Venice AI",
        "https://api.venice.ai/api/v1",
    )
    .with_api_key(key)
}

/// Qwen / Alibaba DashScope.  Reads `DASHSCOPE_API_KEY`.
///
/// Targets the DashScope international endpoint (Singapore).
/// Model: qwen3.6-plus-2026-04-02 — 1M context, 65,536 max output, native tool calling,
/// hybrid thinking (enable_thinking controlled by query layer via provider_options).
///
/// Key agentic parameters:
///   - `enable_thinking`: injected via provider_options when thinking_budget is set
///   - `preserve_thinking`: injected via quirks when FLAG_QWEN_PRESERVE_THINKING
///     is enabled and session heuristics trigger (turn >= 4 or tool_calls >= 3)
///   - `parallel_tool_calls`: enabled (Qwen3.6-Plus supports concurrent tool calls)
///   - `reasoning_content`: the field carrying incremental reasoning tokens
///   - `include_usage_in_stream`: true so token counts are available for budget tracking
///
/// Temperature guidance per Alibaba benchmark docs:
///   - 1.0 for long-horizon / SWE-Bench style tasks
///   - 0.6 for evaluation / structured tasks
///   - We default to 0.6 as a balanced agentic default; callers can override.
pub fn qwen() -> OpenAiCompatProvider {
    let key = std::env::var("DASHSCOPE_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        "qwen",
        "Qwen (Alibaba)",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        // 0.6 balanced default; query layer can bump to 1.0 for long-horizon tasks.
        default_temperature: Some(0.6),
        // DashScope returns incremental reasoning tokens in this field.
        reasoning_field: Some("reasoning_content".to_string()),
        // Required to receive token counts in streaming responses.
        include_usage_in_stream: true,
        // Qwen3.6-Plus supports parallel tool calls; enable for faster agentic loops.
        parallel_tool_calls: Some(true),
        // preserve_thinking is toggled at runtime by the query layer based on
        // FLAG_QWEN_PRESERVE_THINKING and session heuristics; start false here.
        preserve_thinking: false,
        ..Default::default()
    })
}

/// Mistral AI — Reads `MISTRAL_API_KEY`.
/// Uses OpenAI-compatible format with Mistral-specific quirks:
///   - Tool call IDs must be alphanumeric only, truncated to 9 chars and
///     right-padded with zeroes to exactly 9 chars.
///   - An assistant "Done." turn is inserted between tool→user message transitions.
pub fn mistral() -> OpenAiCompatProvider {
    let key = std::env::var("MISTRAL_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::MISTRAL,
        "Mistral AI",
        "https://api.mistral.ai/v1",
    )
    .with_api_key(key)
    .with_quirks(ProviderQuirks {
        tool_id_max_len: Some(9),
        tool_id_alphanumeric_only: true,
        fix_tool_user_sequence: true,
        include_usage_in_stream: true,
        overflow_patterns: vec!["too large for model with".to_string()],
        ..Default::default()
    })
}

/// OpenRouter — unified API gateway to many models.  Reads `OPENROUTER_API_KEY`.
pub fn openrouter() -> OpenAiCompatProvider {
    let key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::OPENROUTER,
        "OpenRouter",
        "https://openrouter.ai/api/v1",
    )
    .with_api_key(key)
    .with_header("HTTP-Referer", "https://mangocode.ai/")
    .with_header("X-Title", "MangoCode")
    .with_quirks(ProviderQuirks {
        include_usage_in_stream: true,
        ..Default::default()
    })
}

/// SambaNova — fast inference cloud.  Reads `SAMBANOVA_API_KEY`.
pub fn sambanova() -> OpenAiCompatProvider {
    let key = std::env::var("SAMBANOVA_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::SAMBANOVA,
        "SambaNova",
        "https://api.sambanova.ai/v1",
    )
    .with_api_key(key)
}

fn huggingface_api_key() -> String {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
        .or_else(|_| std::env::var("HUGGINGFACE_API_KEY"))
        .unwrap_or_default()
}

/// Hugging Face Inference Providers OpenAI-compatible chat endpoint.
/// Reads `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HUGGINGFACE_API_KEY`.
pub fn huggingface() -> OpenAiCompatProvider {
    let key = huggingface_api_key();
    OpenAiCompatProvider::new(
        ProviderId::HUGGINGFACE,
        "Hugging Face",
        "https://router.huggingface.co/v1",
    )
    .with_api_key(key)
}

/// Nvidia NIM — enterprise AI inference.  Reads `NVIDIA_API_KEY`.
pub fn nvidia() -> OpenAiCompatProvider {
    let key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::NVIDIA,
        "Nvidia",
        "https://integrate.api.nvidia.com/v1",
    )
    .with_api_key(key)
}

/// SiliconFlow — DeepSeek / Qwen hosting.  Reads `SILICONFLOW_API_KEY`.
pub fn siliconflow() -> OpenAiCompatProvider {
    let key = std::env::var("SILICONFLOW_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::SILICONFLOW,
        "SiliconFlow",
        "https://api.siliconflow.com/v1",
    )
    .with_api_key(key)
}

/// Moonshot AI / Kimi.  Reads `MOONSHOT_API_KEY`.
pub fn moonshot() -> OpenAiCompatProvider {
    let key = std::env::var("MOONSHOT_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::MOONSHOT,
        "Moonshot AI",
        "https://api.moonshot.ai/v1",
    )
    .with_api_key(key)
}

/// Zhipu AI / GLM.  Reads `ZHIPU_API_KEY`.
pub fn zhipu() -> OpenAiCompatProvider {
    let key = std::env::var("ZHIPU_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::ZHIPU,
        "Zhipu AI",
        "https://open.bigmodel.cn/api/paas/v4",
    )
    .with_api_key(key)
}

/// Nebius — Llama / Qwen hosting.  Reads `NEBIUS_API_KEY`.
pub fn nebius() -> OpenAiCompatProvider {
    let key = std::env::var("NEBIUS_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::NEBIUS,
        "Nebius",
        "https://api.tokenfactory.nebius.com/v1",
    )
    .with_api_key(key)
}

/// Novita — Llama / Stable Diffusion hosting.  Reads `NOVITA_API_KEY`.
pub fn novita() -> OpenAiCompatProvider {
    let key = std::env::var("NOVITA_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::NOVITA,
        "Novita",
        "https://api.novita.ai/v3/openai",
    )
    .with_api_key(key)
}

/// OVHcloud — EU-hosted AI.  Reads `OVHCLOUD_API_KEY`.
pub fn ovhcloud() -> OpenAiCompatProvider {
    let key = std::env::var("OVHCLOUD_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::OVHCLOUD,
        "OVHcloud",
        "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    )
    .with_api_key(key)
}

/// Scaleway — EU cloud AI.  Reads `SCALEWAY_API_KEY`.
pub fn scaleway() -> OpenAiCompatProvider {
    let key = std::env::var("SCALEWAY_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::SCALEWAY,
        "Scaleway",
        "https://api.scaleway.ai/v1",
    )
    .with_api_key(key)
}

/// Vultr — cloud inference.  Reads `VULTR_API_KEY`.
pub fn vultr_ai() -> OpenAiCompatProvider {
    let key = std::env::var("VULTR_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::VULTR,
        "Vultr",
        "https://api.vultrinference.com/v1",
    )
    .with_api_key(key)
}

/// Baseten — model serving.  Reads `BASETEN_API_KEY`.
pub fn baseten() -> OpenAiCompatProvider {
    let key = std::env::var("BASETEN_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::BASETEN,
        "Baseten",
        "https://inference.baseten.co/v1",
    )
    .with_api_key(key)
}

/// Friendli — serverless inference.  Reads `FRIENDLI_TOKEN`.
pub fn friendli() -> OpenAiCompatProvider {
    let key = std::env::var("FRIENDLI_TOKEN").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::FRIENDLI,
        "Friendli",
        "https://api.friendli.ai/serverless/v1",
    )
    .with_api_key(key)
}

/// Upstage — Solar models.  Reads `UPSTAGE_API_KEY`.
pub fn upstage() -> OpenAiCompatProvider {
    let key = std::env::var("UPSTAGE_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::UPSTAGE,
        "Upstage",
        "https://api.upstage.ai/v1/solar",
    )
    .with_api_key(key)
}

/// StepFun — Step models.  Reads `STEPFUN_API_KEY`.
pub fn stepfun() -> OpenAiCompatProvider {
    let key = std::env::var("STEPFUN_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(ProviderId::STEPFUN, "StepFun", "https://api.stepfun.com/v1")
        .with_api_key(key)
}

/// Fireworks AI — fast inference.  Reads `FIREWORKS_API_KEY`.
pub fn fireworks() -> OpenAiCompatProvider {
    let key = std::env::var("FIREWORKS_API_KEY").unwrap_or_default();
    OpenAiCompatProvider::new(
        ProviderId::FIREWORKS,
        "Fireworks AI",
        "https://api.fireworks.ai/inference/v1",
    )
    .with_api_key(key)
}

#[cfg(test)]
mod tests {
    use super::{native_ollama_base_url, normalize_ollama_base_url, parse_ollama_tags_response};

    #[test]
    fn normalize_ollama_accepts_bare_host_port() {
        assert_eq!(
            normalize_ollama_base_url("127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn normalize_ollama_accepts_full_http_url() {
        assert_eq!(
            normalize_ollama_base_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn normalize_ollama_strips_trailing_slash() {
        assert_eq!(
            normalize_ollama_base_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn normalize_ollama_does_not_double_v1_suffix() {
        assert_eq!(
            normalize_ollama_base_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            normalize_ollama_base_url("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn normalize_ollama_keeps_https_scheme() {
        assert_eq!(
            normalize_ollama_base_url("https://ollama.example.com"),
            "https://ollama.example.com/v1"
        );
    }

    #[test]
    fn native_base_url_strips_v1_suffix() {
        assert_eq!(
            native_ollama_base_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn native_base_url_handles_bare_host_port() {
        assert_eq!(
            native_ollama_base_url("127.0.0.1:11434"),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn native_base_url_handles_full_url_without_v1() {
        assert_eq!(
            native_ollama_base_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn native_base_url_strips_trailing_slash_before_v1_check() {
        assert_eq!(
            native_ollama_base_url("http://127.0.0.1:11434/v1/"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            native_ollama_base_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434"
        );
    }

    #[test]
    fn native_base_url_keeps_https() {
        assert_eq!(
            native_ollama_base_url("https://ollama.example.com/v1"),
            "https://ollama.example.com"
        );
    }

    #[test]
    fn parse_tags_response_extracts_names() {
        let body = r#"{
            "models": [
                {
                    "name": "qwen3:8b",
                    "modified_at": "2025-01-01T00:00:00Z",
                    "size": 4920000000,
                    "digest": "abc123",
                    "details": {
                        "family": "qwen",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_K_M"
                    }
                },
                { "name": "llama3.2:latest" }
            ]
        }"#;
        let models = parse_ollama_tags_response(body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "qwen3:8b");
        assert_eq!(models[0].size, Some(4_920_000_000));
        let details = models[0].details.as_ref().expect("details");
        assert_eq!(details.parameter_size.as_deref(), Some("8B"));
        assert_eq!(details.quantization_level.as_deref(), Some("Q4_K_M"));
        assert_eq!(models[1].name, "llama3.2:latest");
    }

    #[test]
    fn parse_tags_response_handles_empty_models() {
        assert!(parse_ollama_tags_response(r#"{"models":[]}"#).is_empty());
    }

    #[test]
    fn parse_tags_response_handles_invalid_json() {
        // Malformed bodies should fall back to an empty list rather than panic.
        assert!(parse_ollama_tags_response("not json").is_empty());
        assert!(parse_ollama_tags_response("").is_empty());
    }

    #[test]
    fn parse_tags_response_handles_tags_field() {
        let body = r#"{
            "tags": [
                { "name": "llama3.2:latest" }
            ]
        }"#;
        let models = parse_ollama_tags_response(body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "llama3.2:latest");
    }

    #[test]
    fn parse_tags_response_handles_array_directly() {
        let body = r#"[
            { "name": "qwen3:8b" },
            { "name": "llama3.2:latest" }
        ]"#;
        let models = parse_ollama_tags_response(body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[1].name, "llama3.2:latest");
    }
}
