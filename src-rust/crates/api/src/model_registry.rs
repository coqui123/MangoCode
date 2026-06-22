// model_registry.rs — Model Registry with bundled snapshot and optional
// models.dev refresh (Phase 3).
//
// The registry is pre-populated with a hardcoded snapshot of popular models
// from Anthropic, OpenAI, and Google.  At runtime callers may optionally call
// `refresh_from_models_dev` to extend/update the registry from the public
// models.dev API.  All network failures are swallowed — the bundled snapshot
// is always sufficient for normal operation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mangocode_core::provider_id::{ModelId, ProviderId};

use crate::provider::ModelInfo;

// ---------------------------------------------------------------------------
// Natural sort helper
// ---------------------------------------------------------------------------

/// Natural comparison for alphanumeric strings (e.g., "qwen3.10" > "qwen3.2")
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        let a_next = a_chars.next();
        let b_next = b_chars.next();

        match (a_next, b_next) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(a_c), Some(b_c)) => {
                if a_c.is_ascii_digit() && b_c.is_ascii_digit() {
                    // Parse numbers
                    let mut a_num = String::new();
                    let mut b_num = String::new();

                    a_num.push(a_c);
                    while let Some(&c) = a_chars.peek() {
                        if c.is_ascii_digit() {
                            if let Some(next) = a_chars.next() {
                                a_num.push(next);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    b_num.push(b_c);
                    while let Some(&c) = b_chars.peek() {
                        if c.is_ascii_digit() {
                            if let Some(next) = b_chars.next() {
                                b_num.push(next);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    match compare_numeric_chunks(&a_num, &b_num) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                } else {
                    match a_c.cmp(&b_c) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
            }
        }
    }
}

fn compare_numeric_chunks(a: &str, b: &str) -> std::cmp::Ordering {
    let a_trimmed = a.trim_start_matches('0');
    let b_trimmed = b.trim_start_matches('0');
    let a_normalized = if a_trimmed.is_empty() { "0" } else { a_trimmed };
    let b_normalized = if b_trimmed.is_empty() { "0" } else { b_trimmed };

    match a_normalized.len().cmp(&b_normalized.len()) {
        std::cmp::Ordering::Equal => a_normalized
            .cmp(b_normalized)
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.cmp(b)),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// ModelEntry
// ---------------------------------------------------------------------------

/// Extended model information with pricing and capability flags.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub info: ModelInfo,
    /// USD per 1M input tokens (`None` = unknown / free).
    pub cost_input: Option<f64>,
    /// USD per 1M output tokens.
    pub cost_output: Option<f64>,
    /// Cache read pricing per 1M tokens.
    pub cost_cache_read: Option<f64>,
    /// Cache write pricing per 1M tokens.
    pub cost_cache_write: Option<f64>,
    /// Supports tool / function calling.
    pub tool_calling: bool,
    /// Supports extended thinking / reasoning.
    pub reasoning: bool,
    /// Supports vision / image input.
    pub vision: bool,
    /// Model family (e.g. `"claude"`, `"gpt"`, `"gemini"`).
    pub family: Option<String>,
    /// Human-readable status: `"active"`, `"beta"`, or `"deprecated"`.
    pub status: String,
}

impl ModelEntry {
    /// Provider-agnostic score for coding-agent suitability.
    ///
    /// This is intentionally computed from existing metadata so older model
    /// cache files remain compatible.  It is used for recommendations and
    /// default selection only; an explicit user-selected model still wins.
    pub fn coding_capability_score(&self) -> u16 {
        let mut score = 0u16;

        if self.tool_calling {
            score += 35;
        }
        if self.reasoning || !self.info.supported_reasoning_levels.is_empty() {
            score += 25;
        }

        score += match self.info.context_window {
            ctx if ctx >= 1_000_000 => 20,
            ctx if ctx >= 200_000 => 16,
            ctx if ctx >= 128_000 => 12,
            ctx if ctx >= 32_000 => 6,
            _ => 0,
        };

        score += match self.info.max_output_tokens {
            out if out >= 32_000 => 10,
            out if out >= 16_000 => 8,
            out if out >= 8_000 => 5,
            _ => 0,
        };

        score += match self.status.as_str() {
            "active" => 5,
            "beta" => 2,
            "deprecated" => 0,
            _ => 1,
        };

        let id = self.info.id.to_ascii_lowercase();
        if id.contains("codex")
            || id.contains("claude")
            || id.contains("gpt")
            || id.contains("gemini")
            || id.contains("qwen")
            || id.contains("deepseek")
        {
            score += 5;
        }

        score.min(100)
    }

    pub fn coding_capability_label(&self) -> &'static str {
        match self.coding_capability_score() {
            85..=100 => "excellent",
            65..=84 => "strong",
            40..=64 => "basic",
            _ => "limited",
        }
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry
// ---------------------------------------------------------------------------

pub struct ModelRegistry {
    /// Keyed by `"provider_id/model_id"`.
    entries: HashMap<String, ModelEntry>,
    /// Optional path for on-disk persistence between sessions.
    cache_path: Option<PathBuf>,
    /// When the registry was last refreshed from the network.
    last_refresh: Option<Instant>,
    /// Minimum age before a network refresh is attempted again.
    refresh_interval: Duration,
}

/// True when the lmstudio backend resolves to the Copilot-pirate proxy, using
/// the same env resolution the provider uses (`LM_STUDIO_HOST`, default
/// `http://localhost:1234`, plus the `MANGOCODE_COPILOT_PIRATE` override).
fn copilot_pirate_lm_studio_active() -> bool {
    let host =
        std::env::var("LM_STUDIO_HOST").unwrap_or_else(|_| "http://localhost:1234".to_string());
    let base = format!("{}/v1", host.trim_end_matches('/'));
    crate::providers::openai_compat::is_copilot_pirate_backend(&base)
}

impl ModelRegistry {
    /// Create a new registry pre-populated with the bundled snapshot.
    pub fn new() -> Self {
        let mut registry = Self {
            entries: HashMap::new(),
            cache_path: None,
            last_refresh: None,
            refresh_interval: Duration::from_secs(5 * 60),
        };
        registry.load_bundled_snapshot();
        registry
    }

    /// Configure a cache file path for persistence between sessions.
    pub fn with_cache_path(mut self, path: PathBuf) -> Self {
        self.cache_path = Some(path);
        self
    }

    // -----------------------------------------------------------------------
    // Bundled snapshot
    // -----------------------------------------------------------------------

    fn load_bundled_snapshot(&mut self) {
        self.add_anthropic_models();
        self.add_openai_models();
        self.add_google_models();
        self.add_qwen_models();
        // Only register the synthetic m365-copilot model when the lmstudio
        // backend actually resolves to the Copilot-pirate proxy. Otherwise it
        // would be the only registered lmstudio model and become the
        // auto-selected default for plain LM Studio users — whose server has no
        // such model, producing a 404 on the first request.
        if copilot_pirate_lm_studio_active() {
            self.add_copilot_pirate_models();
        }
    }

    /// Copilot-pirate (M365 Copilot local proxy) via the lmstudio provider slot.
    fn add_copilot_pirate_models(&mut self) {
        for provider_id in ["lmstudio", ProviderId::LM_STUDIO] {
            self.insert(ModelEntry {
                info: ModelInfo {
                    id: ModelId::new("m365-copilot"),
                    provider_id: ProviderId::new(provider_id),
                    name: "M365 Copilot (Copilot-pirate)".to_string(),
                    context_window: 128_000,
                    max_output_tokens: 8_192,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                },
                cost_input: Some(0.0),
                cost_output: Some(0.0),
                cost_cache_read: None,
                cost_cache_write: None,
                tool_calling: true,
                reasoning: false,
                vision: false,
                family: Some("copilot".to_string()),
                status: "active".to_string(),
            });
        }
    }

    fn add_anthropic_models(&mut self) {
        let pid = ProviderId::new(ProviderId::ANTHROPIC);
        let mut models = [
            (
                "claude-opus-4-6",
                "Claude Opus 4.6",
                200_000u32,
                32_000u32,
                15.0f64,
                75.0f64,
            ),
            (
                "claude-sonnet-4-6",
                "Claude Sonnet 4.6",
                200_000,
                16_000,
                3.0,
                15.0,
            ),
            (
                "claude-haiku-4-5-20251001",
                "Claude Haiku 4.5",
                200_000,
                8_096,
                0.8,
                4.0,
            ),
        ];
        models.sort_by(|a, b| natural_cmp(a.0, b.0));
        for (id, name, ctx, out, cost_in, cost_out) in models {
            self.insert(ModelEntry {
                info: ModelInfo {
                    id: ModelId::new(id),
                    provider_id: pid.clone(),
                    name: name.to_string(),
                    context_window: ctx,
                    max_output_tokens: out,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                },
                cost_input: Some(cost_in),
                cost_output: Some(cost_out),
                cost_cache_read: Some(cost_in * 0.1),
                cost_cache_write: Some(cost_in * 1.25),
                tool_calling: true,
                reasoning: true,
                vision: true,
                family: Some("claude".to_string()),
                status: "active".to_string(),
            });
        }
    }

    fn add_openai_models(&mut self) {
        let pid = ProviderId::new(ProviderId::OPENAI);
        let mut models = [
            (
                "gpt-4o", "GPT-4o", 128_000u32, 16_384u32, 2.5f64, 10.0f64, true, false,
            ),
            (
                "gpt-4o-mini",
                "GPT-4o mini",
                128_000,
                16_384,
                0.15,
                0.6,
                true,
                false,
            ),
            ("o3", "o3", 200_000, 100_000, 10.0, 40.0, true, true),
            ("o4-mini", "o4-mini", 200_000, 100_000, 1.1, 4.4, true, true),
        ];
        models.sort_by(|a, b| natural_cmp(a.0, b.0));
        for (id, name, ctx, out, cost_in, cost_out, tools, reasoning) in models {
            self.insert(ModelEntry {
                info: ModelInfo {
                    id: ModelId::new(id),
                    provider_id: pid.clone(),
                    name: name.to_string(),
                    context_window: ctx,
                    max_output_tokens: out,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                },
                cost_input: Some(cost_in),
                cost_output: Some(cost_out),
                cost_cache_read: None,
                cost_cache_write: None,
                tool_calling: tools,
                reasoning,
                vision: true,
                family: Some("gpt".to_string()),
                status: "active".to_string(),
            });
        }
    }

    fn add_google_models(&mut self) {
        let pid = ProviderId::new(ProviderId::GOOGLE);
        let mut models = [
            (
                "gemini-2.5-pro",
                "Gemini 2.5 Pro",
                1_048_576u32,
                65_536u32,
                1.25f64,
                5.0f64,
            ),
            (
                "gemini-2.5-flash",
                "Gemini 2.5 Flash",
                1_048_576,
                65_536,
                0.15,
                0.6,
            ),
            (
                "gemini-2.0-flash",
                "Gemini 2.0 Flash",
                1_048_576,
                8_192,
                0.1,
                0.4,
            ),
        ];
        models.sort_by(|a, b| natural_cmp(a.0, b.0));
        for (id, name, ctx, out, cost_in, cost_out) in models {
            self.insert(ModelEntry {
                info: ModelInfo {
                    id: ModelId::new(id),
                    provider_id: pid.clone(),
                    name: name.to_string(),
                    context_window: ctx,
                    max_output_tokens: out,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                },
                cost_input: Some(cost_in),
                cost_output: Some(cost_out),
                cost_cache_read: None,
                cost_cache_write: None,
                tool_calling: true,
                reasoning: true,
                vision: true,
                family: Some("gemini".to_string()),
                status: "active".to_string(),
            });
        }
    }

    fn add_qwen_models(&mut self) {
        let pid = ProviderId::new("qwen");

        // Specs sourced from Alibaba Cloud Model Studio docs (April 2026).
        // All Qwen 3.6 models: 1M context, 65,536 max output, native tool calling,
        // hybrid thinking models (enable_thinking + preserve_thinking for agents).
        // DashScope intl endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        // Recommended agentic temperature: 1.0 (long-horizon), 0.6 (eval tasks).
        // Default temperature override of 0.55 applied in ProviderQuirks.
        let mut models = [
            ("qwen3.6-flash", "Qwen 3.6 Flash"),
            ("qwen3.6-max-preview", "Qwen 3.6 Max Preview"),
            ("qwen3.6-27b", "Qwen 3.6 27B"),
            ("qwen3.6-flash-2026-04-16", "Qwen 3.6 Flash (2026-04-16)"),
            ("qwen3.6-35b-a3b", "Qwen 3.6 35B A3B"),
            ("qwen3.6-plus-2026-04-02", "Qwen 3.6 Plus (2026-04-02)"),
            ("qwen3.6-plus", "Qwen 3.6 Plus"),
            (
                "qwen3-vl-235b-a22b-thinking",
                "Qwen 3 VL 235B A22B Thinking",
            ),
            ("qwen3-vl-30b-a3b-thinking", "Qwen 3 VL 30B A3B Thinking"),
            ("qwen3-32b", "Qwen 3 32B"),
            ("qwen3.5-35b-a3b", "Qwen 3.5 35B A3B"),
            (
                "qwen3-coder-480b-a35b-instruct",
                "Qwen 3 Coder 480B A35B Instruct",
            ),
            ("qwen3-coder-plus", "Qwen 3 Coder Plus"),
            ("qwen3-vl-8b-thinking", "Qwen 3 VL 8B Thinking"),
            ("qwen3-max-preview", "Qwen 3 Max Preview"),
            ("qwen3.5-flash-2026-02-23", "Qwen 3.5 Flash (2026-02-23)"),
            ("qwen3-vl-flash-2025-10-15", "Qwen 3 VL Flash (2025-10-15)"),
            ("qwen3-8b", "Qwen 3 8B"),
            ("qwen3-0.6b", "Qwen 3 0.6B"),
            ("qwen3-coder-flash", "Qwen 3 Coder Flash"),
            (
                "qwen3-next-80b-a3b-thinking",
                "Qwen 3 Next 80B A3B Thinking",
            ),
            ("qwen3.5-27b", "Qwen 3.5 27B"),
            ("qwen3-vl-flash", "Qwen 3 VL Flash"),
            ("qwen3-14b", "Qwen 3 14B"),
            ("qwen3-max-2025-09-23", "Qwen 3 Max (2025-09-23)"),
            (
                "qwen3-30b-a3b-instruct-2507",
                "Qwen 3 30B A3B Instruct 2507",
            ),
            (
                "qwen3-235b-a22b-instruct-2507",
                "Qwen 3 235B A22B Instruct 2507",
            ),
            (
                "qwen3-coder-plus-2025-07-22",
                "Qwen 3 Coder Plus (2025-07-22)",
            ),
            ("qwen3.5-plus-2026-04-20", "Qwen 3.5 Plus (2026-04-20)"),
            ("qwen3.5-122b-a10b", "Qwen 3.5 122B A10B"),
            ("qwen3-max", "Qwen 3 Max"),
            ("qwen3.5-plus-2026-02-15", "Qwen 3.5 Plus (2026-02-15)"),
            (
                "qwen3-235b-a22b-thinking-2507",
                "Qwen 3 235B A22B Thinking 2507",
            ),
            ("qwen3.5-397b-a17b", "Qwen 3.5 397B A17B"),
            ("qwen3-vl-plus-2025-09-23", "Qwen 3 VL Plus (2025-09-23)"),
            ("qwen3-coder-next", "Qwen 3 Coder Next"),
            ("qwen3.5-flash", "Qwen 3.5 Flash"),
            (
                "qwen3-30b-a3b-thinking-2507",
                "Qwen 3 30B A3B Thinking 2507",
            ),
            (
                "qwen3-coder-plus-2025-09-23",
                "Qwen 3 Coder Plus (2025-09-23)",
            ),
            ("qwen3-max-2026-01-23", "Qwen 3 Max (2026-01-23)"),
            ("qwen3-vl-flash-2026-01-22", "Qwen 3 VL Flash (2026-01-22)"),
            ("qwen3-vl-30b-a3b-instruct", "Qwen 3 VL 30B A3B Instruct"),
            (
                "qwen3-coder-30b-a3b-instruct",
                "Qwen 3 Coder 30B A3B Instruct",
            ),
            (
                "qwen3-vl-235b-a22b-instruct",
                "Qwen 3 VL 235B A22B Instruct",
            ),
            ("qwen3-4b", "Qwen 3 4B"),
            ("qwen3-235b-a22b", "Qwen 3 235B A22B"),
            ("qwen3-1.7b", "Qwen 3 1.7B"),
            ("qwen3-vl-plus", "Qwen 3 VL Plus"),
            ("qwen3-30b-a3b", "Qwen 3 30B A3B"),
            ("qwen3-vl-8b-instruct", "Qwen 3 VL 8B Instruct"),
            (
                "qwen3-coder-flash-2025-07-28",
                "Qwen 3 Coder Flash (2025-07-28)",
            ),
            ("qwen3-vl-plus-2025-12-19", "Qwen 3 VL Plus (2025-12-19)"),
            ("qwen3.5-plus", "Qwen 3.5 Plus"),
            (
                "qwen3-next-80b-a3b-instruct",
                "Qwen 3 Next 80B A3B Instruct",
            ),
            ("qvq-max-2025-03-25", "QVQ Max (2025-03-25)"),
            ("qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B Instruct"),
            ("qwen-vl-plus-2025-05-07", "Qwen VL Plus (2025-05-07)"),
            ("qwen-plus-2025-07-28", "Qwen Plus (2025-07-28)"),
            ("qwen-vl-plus-latest", "Qwen VL Plus Latest"),
            ("qwen2.5-vl-3b-instruct", "Qwen 2.5 VL 3B Instruct"),
            ("qwen-max", "Qwen Max"),
            ("qwen2.5-14b-instruct", "Qwen 2.5 14B Instruct"),
            ("qwen-mt-flash", "Qwen MT Flash"),
            ("qwen-vl-max-2025-08-13", "Qwen VL Max (2025-08-13)"),
            ("qwen-max-2025-01-25", "Qwen Max (2025-01-25)"),
            ("qwen2.5-14b-instruct-1m", "Qwen 2.5 14B Instruct 1M"),
            ("qwen-plus", "Qwen Plus"),
            ("qwen-turbo", "Qwen Turbo"),
            ("qvq-max", "QVQ Max"),
            ("qwen-vl-plus-2025-08-15", "Qwen VL Plus (2025-08-15)"),
            ("qwen-vl-max-latest", "Qwen VL Max Latest"),
            (
                "qwen3-next-80b-a3b-thinking",
                "Qwen 3 Next 80B A3B Thinking",
            ),
            ("qwen-turbo-latest", "Qwen Turbo Latest"),
            ("qwen2.5-32b-instruct", "Qwen 2.5 32B Instruct"),
            ("qwen-plus-character", "Qwen Plus Character"),
            ("qwen-flash-character", "Qwen Flash Character"),
            ("qvq-max-latest", "QVQ Max Latest"),
            ("qwen-flash", "Qwen Flash"),
            ("qwen-flash-2025-07-28", "Qwen Flash (2025-07-28)"),
            ("qwen-vl-ocr", "Qwen VL OCR"),
            ("qwen-vl-ocr-2025-11-20", "Qwen VL OCR (2025-11-20)"),
            ("qwen-vl-max-2025-04-08", "Qwen VL Max (2025-04-08)"),
            ("qwen2.5-7b-instruct-1m", "Qwen 2.5 7B Instruct 1M"),
            ("qwen2.5-vl-7b-instruct", "Qwen 2.5 VL 7B Instruct"),
            ("qwen2.5-72b-instruct", "Qwen 2.5 72B Instruct"),
            ("qwen-plus-latest", "Qwen Plus Latest"),
            ("qwen-plus-2025-09-11", "Qwen Plus (2025-09-11)"),
            ("wan2.2-kf2v-flash", "Wan 2.2 KF2V Flash"),
            ("qwen-mt-lite", "Qwen MT Lite"),
            ("qwen-vl-plus-2025-01-25", "Qwen VL Plus (2025-01-25)"),
            ("qwen-turbo-2025-04-28", "Qwen Turbo (2025-04-28)"),
            ("qwen2.5-vl-32b-instruct", "Qwen 2.5 VL 32B Instruct"),
            ("qwen-mt-plus", "Qwen MT Plus"),
            ("qwen-plus-2025-04-28", "Qwen Plus (2025-04-28)"),
            ("qwen-mt-turbo", "Qwen MT Turbo"),
            ("qwen-plus-2025-07-14", "Qwen Plus (2025-07-14)"),
            ("qwq-plus", "QWQ Plus"),
        ];
        models.sort_by(|a, b| natural_cmp(a.0, b.0));
        for (id, name) in models {
            self.insert(ModelEntry {
                info: ModelInfo {
                    id: ModelId::new(id),
                    provider_id: pid.clone(),
                    name: name.to_string(),
                    context_window: 1_000_000u32,
                    max_output_tokens: 65_536u32,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                },
                cost_input: None,
                cost_output: None,
                cost_cache_read: None,
                cost_cache_write: None,
                tool_calling: true,
                reasoning: true,
                vision: true,
                family: Some("qwen".to_string()),
                status: "active".to_string(),
            });
        }
    }

    fn insert(&mut self, entry: ModelEntry) {
        let key = format!("{}/{}", entry.info.provider_id, entry.info.id);
        self.entries.insert(key, entry);
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Get an entry by `"provider_id/model_id"` key.
    pub fn get(&self, provider_id: &str, model_id: &str) -> Option<&ModelEntry> {
        let key = format!("{}/{}", provider_id, model_id);
        self.entries.get(&key)
    }

    /// Resolve a model string into `(ProviderId, ModelId)`.
    ///
    /// Accepts either `"provider/model"` or a bare model name (which defaults
    /// to the Anthropic provider).
    pub fn resolve(s: &str) -> (ProviderId, ModelId) {
        if let Some((provider, model)) = s.split_once('/') {
            (ProviderId::new(provider), ModelId::new(model))
        } else {
            (ProviderId::new(ProviderId::ANTHROPIC), ModelId::new(s))
        }
    }

    /// Look up a bare model name across all registry entries and return the
    /// provider that owns it.  Returns `None` if the model is not found or
    /// if the model string already contains a `"provider/"` prefix.
    ///
    /// This enables automatic provider detection for model names like
    /// `"gemini-3-flash-preview"` → `google`, `"gpt-4o"` → `openai`, etc.
    pub fn find_provider_for_model(&self, model_name: &str) -> Option<ProviderId> {
        // If the caller already has a "provider/model" string, don't search.
        if model_name.contains('/') {
            return None;
        }

        // 1. Family-based heuristic FIRST: well-known model name prefixes
        //    always map to their canonical provider.  This prevents
        //    gateway/proxy entries in the models.dev cache (e.g. "llmgateway")
        //    from hijacking well-known models like claude-* or gpt-*.
        let canonical = if model_name.starts_with("claude") {
            Some(ProviderId::ANTHROPIC)
        } else if model_name.starts_with("gpt-")
            || model_name.starts_with("o1")
            || model_name.starts_with("o3")
            || model_name.starts_with("o4")
        {
            Some(ProviderId::OPENAI)
        } else if model_name.starts_with("gemini") || model_name.starts_with("gemma") {
            Some(ProviderId::GOOGLE)
        } else if model_name.starts_with("deepseek") {
            Some("deepseek")
        } else if model_name.starts_with("mistral")
            || model_name.starts_with("codestral")
            || model_name.starts_with("pixtral")
        {
            Some("mistral")
        } else if model_name.starts_with("grok") {
            Some("xai")
        } else if model_name.starts_with("command-r") || model_name.starts_with("command-a") {
            Some("cohere")
        } else if model_name.starts_with("sonar") {
            Some("perplexity")
        } else {
            None
        };
        if let Some(pid) = canonical {
            return Some(ProviderId::new(pid));
        }

        // 2. Exact match: look for any entry whose model ID matches.
        for entry in self.entries.values() {
            if &*entry.info.id == model_name {
                return Some(entry.info.provider_id.clone());
            }
        }

        // 3. Prefix match: some models have version suffixes that differ from
        // the canonical ID (e.g. "gemini-3-flash-preview" may be stored as
        // "gemini-3-flash-preview-05-20").  Try a prefix match.
        for entry in self.entries.values() {
            if entry.info.id.starts_with(model_name) || model_name.starts_with(&entry.info.id[..]) {
                return Some(entry.info.provider_id.clone());
            }
        }

        None
    }

    /// List all models for a given provider.
    pub fn list_by_provider(&self, provider_id: &str) -> Vec<&ModelEntry> {
        let mut entries: Vec<&ModelEntry> = self
            .entries
            .values()
            .filter(|e| &*e.info.provider_id == provider_id)
            .collect();
        entries.sort_by(|a, b| natural_cmp(&a.info.id, &b.info.id));
        entries
    }

    /// Pick the best default model for a given provider.
    ///
    /// Uses a priority-based scoring system inspired by OpenCode:
    ///   1. Models matching well-known "flagship" patterns rank highest
    ///   2. Coding-agent capability score prefers tool/reasoning/context fit
    ///   3. Models with "latest" in the ID are preferred
    ///   4. Otherwise, models are ranked by descending ID (newer versions first)
    ///
    /// Returns the model ID string, or `None` if the provider has no models.
    pub fn best_model_for_provider(&self, provider_id: &str) -> Option<String> {
        let mut models: Vec<&ModelEntry> = self.list_by_provider(provider_id);
        if models.is_empty() {
            return None;
        }

        // Priority patterns: models matching these substrings are considered
        // "flagship" quality and are preferred as defaults.
        // Priority list similar to OpenCode defaults ("gpt-5", flagship Claude, etc.)
        let priority_patterns: &[&str] = &[
            "claude-sonnet-4",
            "gpt-5",
            "gpt-4o",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "deepseek-chat",
            "mistral-large",
            "grok-2",
            "command-r-plus",
            "llama-3.3-70b",
            "sonar-pro",
        ];

        // Score each model: lower is better.
        // Priority match = index in priority_patterns (or usize::MAX if not found).
        // "latest" suffix bonus = 0, otherwise 1.
        // Tie-break: descending ID.
        models.sort_by(|a, b| {
            let id_a: &str = &a.info.id;
            let id_b: &str = &b.info.id;

            let prio_a = priority_patterns
                .iter()
                .position(|pat| id_a.contains(pat))
                .unwrap_or(usize::MAX);
            let prio_b = priority_patterns
                .iter()
                .position(|pat| id_b.contains(pat))
                .unwrap_or(usize::MAX);

            prio_a
                .cmp(&prio_b)
                .then_with(|| {
                    let latest_a = if id_a.contains("latest") { 0u8 } else { 1 };
                    let latest_b = if id_b.contains("latest") { 0u8 } else { 1 };
                    latest_a.cmp(&latest_b)
                })
                .then_with(|| {
                    b.coding_capability_score()
                        .cmp(&a.coding_capability_score())
                })
                .then_with(|| id_b.cmp(id_a)) // descending by ID
        });

        models.first().map(|e| e.info.id.to_string())
    }

    /// Pick the best "small" model for a given provider.
    ///
    /// Small models are optimised for speed and cost rather than quality.
    /// Uses the same priority-sort pattern as [`best_model_for_provider`]
    /// but with a different priority list targeting lightweight models.
    pub fn best_small_model_for_provider(&self, provider_id: &str) -> Option<String> {
        let mut models: Vec<&ModelEntry> = self.list_by_provider(provider_id);
        if models.is_empty() {
            return None;
        }

        let small_priority: &[&str] = &[
            "claude-haiku-4",
            "gpt-4o-mini",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "deepseek-chat",
            "mistral-small",
            "grok-2-mini",
            "command-r",
            "llama-3.3-8b",
            "sonar",
        ];

        models.sort_by(|a, b| {
            let id_a: &str = &a.info.id;
            let id_b: &str = &b.info.id;

            let prio_a = small_priority
                .iter()
                .position(|pat| id_a.contains(pat))
                .unwrap_or(usize::MAX);
            let prio_b = small_priority
                .iter()
                .position(|pat| id_b.contains(pat))
                .unwrap_or(usize::MAX);

            prio_a
                .cmp(&prio_b)
                .then_with(|| {
                    let latest_a = if id_a.contains("latest") { 0u8 } else { 1 };
                    let latest_b = if id_b.contains("latest") { 0u8 } else { 1 };
                    latest_a.cmp(&latest_b)
                })
                .then_with(|| id_b.cmp(id_a))
        });

        models.first().map(|e| e.info.id.to_string())
    }

    /// List every entry in the registry.
    pub fn list_all(&self) -> Vec<&ModelEntry> {
        let mut entries: Vec<&ModelEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| {
            (*a.info.provider_id)
                .cmp(&*b.info.provider_id)
                .then_with(|| natural_cmp(&a.info.id, &b.info.id))
        });
        entries
    }

    // -----------------------------------------------------------------------
    // Network refresh
    // -----------------------------------------------------------------------

    /// Attempt to refresh the registry from the models.dev public API.
    ///
    /// Returns `Ok(true)` if new data was fetched, `Ok(false)` if the cache
    /// was still fresh.  All network or parse failures are silenced — the
    /// bundled snapshot is always sufficient.
    pub async fn refresh_from_models_dev(&mut self) -> anyhow::Result<bool> {
        if let Some(last) = self.last_refresh {
            if last.elapsed() < self.refresh_interval {
                return Ok(false);
            }
        }

        let url = std::env::var("MODELS_DEV_URL")
            .unwrap_or_else(|_| "https://models.dev/api.json".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let resp = client.get(&url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let json: serde_json::Value = r.json().await?;
                self.parse_models_dev_response(&json);
                self.last_refresh = Some(Instant::now());
                if let Some(ref path) = self.cache_path.clone() {
                    self.save_cache(path);
                }
                Ok(true)
            }
            // Fail silently — bundled snapshot is sufficient.
            _ => Ok(false),
        }
    }

    fn parse_models_dev_response(&mut self, json: &serde_json::Value) {
        // models.dev format:
        // { "provider_id": { "models": { "model_id": { "name": "...", "limit": {...}, "cost": {...} } } } }
        if let Some(obj) = json.as_object() {
            for (provider_id, provider_data) in obj {
                if let Some(models) = provider_data.get("models").and_then(|m| m.as_object()) {
                    for (model_id, model_data) in models {
                        let ctx = model_data
                            .get("limit")
                            .and_then(|l| l.get("context"))
                            .and_then(|c| c.as_u64())
                            .unwrap_or(4096) as u32;
                        let out = model_data
                            .get("limit")
                            .and_then(|l| l.get("output"))
                            .and_then(|o| o.as_u64())
                            .unwrap_or(4096) as u32;
                        let cost_in = model_data
                            .get("cost")
                            .and_then(|c| c.get("input"))
                            .and_then(|i| i.as_f64());
                        let cost_out = model_data
                            .get("cost")
                            .and_then(|c| c.get("output"))
                            .and_then(|o| o.as_f64());
                        let name = model_data
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or(model_id)
                            .to_string();
                        let tool_calling = model_data
                            .get("tool_call")
                            .and_then(|t| t.as_bool())
                            .unwrap_or(false);
                        let reasoning = model_data
                            .get("reasoning")
                            .and_then(|r| r.as_bool())
                            .unwrap_or(false);

                        let pid = ProviderId::new(provider_id.as_str());
                        let mid = ModelId::new(model_id.as_str());
                        let key = format!("{}/{}", pid, mid);

                        // models.dev is the source of truth — overwrite bundled snapshot data.
                        self.entries.insert(
                            key,
                            ModelEntry {
                                info: ModelInfo {
                                    id: mid,
                                    provider_id: pid,
                                    name,
                                    context_window: ctx,
                                    max_output_tokens: out,
                                    default_reasoning_level: None,
                                    supported_reasoning_levels: Vec::new(),
                                },
                                cost_input: cost_in,
                                cost_output: cost_out,
                                cost_cache_read: None,
                                cost_cache_write: None,
                                tool_calling,
                                reasoning,
                                vision: false,
                                family: None,
                                status: "active".to_string(),
                            },
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cache persistence
    // -----------------------------------------------------------------------

    fn save_cache(&self, path: &PathBuf) {
        let json = match serde_json::to_string_pretty(&self.entries) {
            Ok(json) => json,
            Err(err) => {
                tracing::warn!(error = %err, "failed to serialize model registry cache");
                return;
            }
        };
        if let Err(err) = std::fs::write(path, json) {
            tracing::warn!(
                error = %err,
                path = %path.display(),
                "failed to write model registry cache"
            );
        }
    }

    /// Load a previously saved cache file, merging entries into the registry.
    ///
    /// The cache file may be either:
    /// 1. The raw models.dev `api.json` response (providers at the top level), or
    /// 2. Our own serialized `HashMap<String, ModelEntry>` format.
    ///
    /// Both formats are tried in order so the background fetch can simply save
    /// the raw models.dev response to disk and this method will ingest it.
    pub fn load_cache(&mut self, path: &PathBuf) {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => return,
        };
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            // Heuristic: if the top-level object contains a key whose value has
            // a "models" sub-object, it's the raw models.dev format.
            let looks_like_models_dev = json
                .as_object()
                .map(|obj| obj.values().any(|v| v.get("models").is_some()))
                .unwrap_or(false);

            if looks_like_models_dev {
                self.parse_models_dev_response(&json);
                return;
            }
        }
        // Fall back to our own serialized format.
        if let Ok(entries) = serde_json::from_str::<HashMap<String, ModelEntry>>(&data) {
            self.entries.extend(entries);
        }
    }

    /// Load the standard MangoCode models.dev cache from the user cache dir.
    ///
    /// `models.json` is the current cache name used by the CLI/TUI fetch path.
    /// `models_dev.json` is retained as a fallback for older installs.
    pub fn load_standard_cache(&mut self) {
        let Some(cache_dir) = dirs::cache_dir() else {
            return;
        };
        let cache_dir = cache_dir.join("mangocode");
        let primary = cache_dir.join("models.json");
        if primary.exists() {
            self.load_cache(&primary);
            return;
        }
        self.load_cache(&cache_dir.join("models_dev.json"));
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Dynamic model resolution helper
// ---------------------------------------------------------------------------

/// Resolve the effective model for a [`Config`], using the model registry to
/// dynamically pick the best available model for the active provider.
///
/// **Resolution order** (same general approach as OpenCode):
///  1. If the user explicitly set `config.model`, use it verbatim.
///  2. Consult the model registry for the configured provider's best model
///     (scored by flagship priority → "latest" preference → ID desc).
///  3. Fall back to the hardcoded table in [`Config::effective_model()`].
pub fn effective_model_for_config(
    config: &mangocode_core::Config,
    registry: &ModelRegistry,
) -> String {
    // Explicit user override — always wins when it contains a real model ID.
    if config
        .model
        .as_deref()
        .is_some_and(|model| !model.trim().is_empty())
    {
        return config.effective_model().to_string();
    }

    // Try the model registry for the configured provider.
    if let Some(provider_id) = config.provider.as_deref() {
        if let Some(best) = registry.best_model_for_provider(provider_id) {
            return best;
        }
    }

    // Fall back to the hardcoded table.
    config.effective_model().to_string()
}

/// Normalize a user-entered model ID.
pub fn normalize_model_id(model: &str) -> Option<String> {
    let model = model.trim();
    (!model.is_empty()).then(|| model.to_string())
}

fn infer_provider_for_model_selection(
    model: &str,
    registry: Option<&ModelRegistry>,
) -> Option<String> {
    if let Some((provider, _)) = mangocode_core::ProviderId::split_known_model_prefix(model) {
        return Some(provider.to_string());
    }

    if let Some(provider) = registry.and_then(|registry| registry.find_provider_for_model(model)) {
        return Some(provider.to_string());
    }

    let mut fallback = ModelRegistry::new();
    fallback.load_standard_cache();
    fallback
        .find_provider_for_model(model)
        .map(|provider| provider.to_string())
}

fn is_codex_provider_id(provider: &str) -> bool {
    matches!(provider, ProviderId::OPENAI_CODEX | "codex")
}

/// Apply a user-selected model to config, keeping provider/model wiring in sync.
pub fn apply_model_selection_to_config(
    config: &mut mangocode_core::Config,
    model: &str,
    registry: Option<&ModelRegistry>,
) -> bool {
    let Some(model) = normalize_model_id(model) else {
        return false;
    };

    if let Some((provider, model_id)) = mangocode_core::ProviderId::split_known_model_prefix(&model)
        .map(|(provider, model_id)| (provider.to_string(), model_id.to_string()))
    {
        let canonical_provider = if provider == "codex" {
            ProviderId::OPENAI_CODEX.to_string()
        } else {
            provider
        };
        config.provider = Some(canonical_provider.clone());
        config.model = Some(
            if canonical_provider == mangocode_core::ProviderId::ANTHROPIC {
                model_id
            } else if canonical_provider == ProviderId::OPENAI_CODEX {
                format!("{}/{}", ProviderId::OPENAI_CODEX, model_id)
            } else {
                model
            },
        );
        return true;
    }

    if config
        .provider
        .as_deref()
        .filter(|provider| is_codex_provider_id(provider))
        .is_some()
        && mangocode_core::codex_oauth::is_bare_codex_model_alias(&model)
    {
        config.provider = Some(ProviderId::OPENAI_CODEX.to_string());
        config.model = Some(model);
        return true;
    }

    if let Some(provider) = infer_provider_for_model_selection(&model, registry) {
        config.provider = Some(provider);
    }
    config.model = Some(model);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entry(provider: &str, model: &str) -> ModelEntry {
        ModelEntry {
            info: ModelInfo {
                id: ModelId::new(model),
                provider_id: ProviderId::new(provider),
                name: model.to_string(),
                context_window: 1000,
                max_output_tokens: 100,
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
            },
            cost_input: None,
            cost_output: None,
            cost_cache_read: None,
            cost_cache_write: None,
            tool_calling: false,
            reasoning: false,
            vision: false,
            family: None,
            status: "active".to_string(),
        }
    }

    #[test]
    fn model_selection_trims_and_rejects_blank_models() {
        let mut config = mangocode_core::Config {
            provider: Some("anthropic".to_string()),
            model: Some("claude-haiku-4-5".to_string()),
            ..Default::default()
        };

        assert!(!apply_model_selection_to_config(&mut config, "   ", None));
        assert_eq!(config.provider.as_deref(), Some("anthropic"));
        assert_eq!(config.model.as_deref(), Some("claude-haiku-4-5"));

        assert!(apply_model_selection_to_config(
            &mut config,
            " openai/gpt-4o ",
            None
        ));
        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn model_selection_strips_anthropic_prefix() {
        let mut config = mangocode_core::Config::default();

        assert!(apply_model_selection_to_config(
            &mut config,
            "anthropic/claude-haiku-4-5",
            None
        ));

        assert_eq!(config.provider.as_deref(), Some("anthropic"));
        assert_eq!(config.model.as_deref(), Some("claude-haiku-4-5"));
    }

    #[test]
    fn model_selection_preserves_configured_codex_for_bare_codex_models() {
        let mut config = mangocode_core::Config {
            provider: Some("openai-codex".to_string()),
            ..Default::default()
        };

        assert!(apply_model_selection_to_config(
            &mut config,
            "gpt-5.5",
            None
        ));

        assert_eq!(config.provider.as_deref(), Some("openai-codex"));
        assert_eq!(config.model.as_deref(), Some("gpt-5.5"));
    }

    #[test]
    fn model_selection_normalizes_codex_provider_alias_for_bare_codex_models() {
        let mut config = mangocode_core::Config {
            provider: Some("codex".to_string()),
            ..Default::default()
        };

        assert!(apply_model_selection_to_config(
            &mut config,
            "gpt-5-codex",
            None
        ));

        assert_eq!(config.provider.as_deref(), Some("openai-codex"));
        assert_eq!(config.model.as_deref(), Some("gpt-5-codex"));
    }

    #[test]
    fn model_selection_normalizes_codex_provider_prefix_alias() {
        let mut config = mangocode_core::Config::default();

        assert!(apply_model_selection_to_config(
            &mut config,
            "codex/gpt-5.5",
            None
        ));

        assert_eq!(config.provider.as_deref(), Some("openai-codex"));
        assert_eq!(config.model.as_deref(), Some("openai-codex/gpt-5.5"));
    }

    #[test]
    fn model_selection_configured_codex_does_not_claim_generic_openai_model() {
        let mut config = mangocode_core::Config {
            provider: Some("openai-codex".to_string()),
            ..Default::default()
        };

        assert!(apply_model_selection_to_config(&mut config, "gpt-4o", None));

        assert_eq!(config.provider.as_deref(), Some("openai"));
        assert_eq!(config.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn natural_sort_orders_multi_digit_model_suffixes() {
        let mut models = ["qwen3.10", "qwen3.2", "qwen3.1"];
        models.sort_by(|a, b| natural_cmp(a, b));

        assert_eq!(models, ["qwen3.1", "qwen3.2", "qwen3.10"]);
    }

    #[test]
    fn natural_sort_handles_numeric_suffixes_larger_than_u64() {
        let mut models = [
            "model-999999999999999999999999999999",
            "model-1000000000000000000000000000000",
            "model-2",
        ];
        models.sort_by(|a, b| natural_cmp(a, b));

        assert_eq!(
            models,
            [
                "model-2",
                "model-999999999999999999999999999999",
                "model-1000000000000000000000000000000",
            ]
        );
    }

    #[test]
    fn natural_sort_breaks_ties_for_leading_zero_numeric_chunks() {
        let mut models = ["model-001", "model-1", "model-01"];
        models.sort_by(|a, b| natural_cmp(a, b));

        assert_eq!(models, ["model-1", "model-01", "model-001"]);
    }

    #[test]
    fn list_by_provider_returns_stable_natural_order() {
        let mut registry = ModelRegistry::new();
        for id in ["model-10", "model-2", "model-1"] {
            registry.insert(test_entry("sort-test", id));
        }

        let models: Vec<String> = registry
            .list_by_provider("sort-test")
            .iter()
            .map(|entry| entry.info.id.to_string())
            .collect();

        assert_eq!(models, ["model-1", "model-2", "model-10"]);
    }

    #[test]
    fn list_all_returns_stable_provider_then_natural_model_order() {
        let mut registry = ModelRegistry::new();
        for (provider, id) in [
            ("sort-b", "model-10"),
            ("sort-a", "model-10"),
            ("sort-a", "model-2"),
        ] {
            registry.insert(test_entry(provider, id));
        }

        let models: Vec<String> = registry
            .list_all()
            .into_iter()
            .filter(|entry| entry.info.provider_id.starts_with("sort-"))
            .map(|entry| format!("{}/{}", entry.info.provider_id, entry.info.id))
            .collect();

        assert_eq!(
            models,
            ["sort-a/model-2", "sort-a/model-10", "sort-b/model-10"]
        );
    }

    #[test]
    fn coding_capability_score_rewards_agent_relevant_model_traits() {
        let mut limited = test_entry("score-test", "plain-model");
        limited.info.context_window = 16_000;
        limited.info.max_output_tokens = 4_096;
        limited.tool_calling = false;
        limited.reasoning = false;

        let mut strong = test_entry("score-test", "gpt-coding-model");
        strong.info.context_window = 200_000;
        strong.info.max_output_tokens = 32_000;
        strong.tool_calling = true;
        strong.reasoning = true;

        assert!(strong.coding_capability_score() > limited.coding_capability_score());
        assert_eq!(strong.coding_capability_label(), "excellent");
        assert_eq!(limited.coding_capability_label(), "limited");
    }

    #[test]
    fn best_model_uses_coding_capability_as_tie_breaker() {
        let mut registry = ModelRegistry::new();

        let mut weak = test_entry("coding-sort", "alpha-latest");
        weak.info.context_window = 8_000;
        weak.info.max_output_tokens = 4_096;
        weak.tool_calling = false;
        weak.reasoning = false;

        let mut strong = test_entry("coding-sort", "alpha-stable");
        strong.info.context_window = 200_000;
        strong.info.max_output_tokens = 32_000;
        strong.tool_calling = true;
        strong.reasoning = true;

        registry.insert(weak);
        registry.insert(strong);

        // "latest" in the ID is preferred first; capability score breaks ties
        // only among models that are equally "latest" or not.
        assert_eq!(
            registry.best_model_for_provider("coding-sort").as_deref(),
            Some("alpha-latest")
        );
    }
}
