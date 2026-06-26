// provider_id.rs — Branded newtypes for provider and model identifiers.
//
// ProviderId and ModelId are separate newtype wrappers around String so that
// the type system prevents accidentally passing a model name where a provider
// name is expected (and vice-versa).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Deref;

// ---------------------------------------------------------------------------
// ProviderId
// ---------------------------------------------------------------------------

/// A branded identifier for an LLM provider (e.g. "anthropic", "openai").
///
/// Well-known constants are provided as associated constants so callers do
/// not need to hard-code raw strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    /// Construct a `ProviderId` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        ProviderId(s.into())
    }

    // -----------------------------------------------------------------------
    // Well-known provider constants
    // -----------------------------------------------------------------------

    pub const ANTHROPIC: &'static str = "anthropic";
    pub const OPENAI: &'static str = "openai";
    pub const GOOGLE: &'static str = "google";
    pub const GOOGLE_VERTEX: &'static str = "google-vertex";
    pub const AMAZON_BEDROCK: &'static str = "amazon-bedrock";
    pub const AZURE: &'static str = "azure";
    pub const GITHUB_COPILOT: &'static str = "github-copilot";
    pub const MISTRAL: &'static str = "mistral";
    pub const XAI: &'static str = "xai";
    pub const GROQ: &'static str = "groq";
    pub const DEEPINFRA: &'static str = "deepinfra";
    pub const CEREBRAS: &'static str = "cerebras";
    pub const COHERE: &'static str = "cohere";
    pub const TOGETHER_AI: &'static str = "together-ai";
    pub const PERPLEXITY: &'static str = "perplexity";
    pub const OPENROUTER: &'static str = "openrouter";
    pub const OLLAMA: &'static str = "ollama";
    pub const LM_STUDIO: &'static str = "lm-studio";
    pub const LLAMA_CPP: &'static str = "llama-cpp";
    pub const DEEPSEEK: &'static str = "deepseek";
    pub const VENICE: &'static str = "venice";
    pub const SAMBANOVA: &'static str = "sambanova";
    pub const HUGGINGFACE: &'static str = "huggingface";
    pub const NVIDIA: &'static str = "nvidia";
    pub const SILICONFLOW: &'static str = "siliconflow";
    pub const MOONSHOT: &'static str = "moonshotai";
    pub const ZHIPU: &'static str = "zhipuai";
    pub const NEBIUS: &'static str = "nebius";
    pub const OVHCLOUD: &'static str = "ovhcloud";
    pub const SCALEWAY: &'static str = "scaleway";
    pub const VULTR: &'static str = "vultr";
    pub const BASETEN: &'static str = "baseten";
    pub const FRIENDLI: &'static str = "friendli";
    pub const UPSTAGE: &'static str = "upstage";
    pub const STEPFUN: &'static str = "stepfun";
    pub const FIREWORKS: &'static str = "fireworks";
    pub const NOVITA: &'static str = "novita";
    pub const MINIMAX: &'static str = "minimax";
    pub const ANTHROPIC_MAX: &'static str = "anthropic-max";
    /// OpenAI Codex (ChatGPT plan) via OAuth — distinct from API-key `openai`.
    pub const OPENAI_CODEX: &'static str = "openai-codex";

    /// Provider IDs that may be used as the first segment in a canonical
    /// `"provider/model"` model string.
    pub const KNOWN_MODEL_PREFIXES: &'static [&'static str] = &[
        Self::ANTHROPIC,
        Self::ANTHROPIC_MAX,
        Self::OPENAI,
        Self::OPENAI_CODEX,
        "codex",
        Self::GOOGLE,
        Self::GOOGLE_VERTEX,
        Self::GROQ,
        Self::MISTRAL,
        Self::DEEPSEEK,
        Self::XAI,
        Self::COHERE,
        Self::PERPLEXITY,
        Self::CEREBRAS,
        Self::OPENROUTER,
        "togetherai",
        Self::TOGETHER_AI,
        Self::DEEPINFRA,
        Self::VENICE,
        Self::GITHUB_COPILOT,
        Self::OLLAMA,
        "lmstudio",
        Self::LM_STUDIO,
        "llamacpp",
        Self::LLAMA_CPP,
        Self::AZURE,
        Self::AMAZON_BEDROCK,
        Self::HUGGINGFACE,
        Self::NVIDIA,
        Self::FIREWORKS,
        Self::SAMBANOVA,
        Self::MINIMAX,
        Self::SILICONFLOW,
        Self::MOONSHOT,
        "moonshot",
        Self::ZHIPU,
        "zhipu",
        "qwen",
        Self::NEBIUS,
        Self::NOVITA,
        Self::OVHCLOUD,
        Self::SCALEWAY,
        Self::VULTR,
        Self::BASETEN,
        Self::FRIENDLI,
        Self::UPSTAGE,
        Self::STEPFUN,
    ];

    /// Return true when `provider` is a known MangoCode provider identifier
    /// suitable for canonical `"provider/model"` strings.
    pub fn is_known_model_prefix(provider: &str) -> bool {
        Self::KNOWN_MODEL_PREFIXES.contains(&provider)
    }

    /// Split a canonical `"provider/model"` string only when the first segment
    /// is a known provider. Unknown first segments are model namespaces, such as
    /// `"meta-llama/Llama-3"` routed through OpenRouter or another gateway.
    pub fn split_known_model_prefix(model: &str) -> Option<(&str, &str)> {
        let (provider, model_id) = model.split_once('/')?;
        if !model_id.is_empty() && Self::is_known_model_prefix(provider) {
            Some((provider, model_id))
        } else {
            None
        }
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Deref for ProviderId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for ProviderId {
    fn from(s: String) -> Self {
        ProviderId(s)
    }
}

impl From<&str> for ProviderId {
    fn from(s: &str) -> Self {
        ProviderId(s.to_string())
    }
}

impl PartialEq<str> for ProviderId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ProviderId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

// ---------------------------------------------------------------------------
// ModelId
// ---------------------------------------------------------------------------

/// A branded identifier for a model (e.g. "claude-opus-4-5", "gpt-4o").
///
/// Kept separate from `ProviderId` for type safety — you cannot accidentally
/// pass a model name where a provider name is expected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    /// Construct a `ModelId` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        ModelId(s.into())
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Deref for ModelId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        ModelId(s)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        ModelId(s.to_string())
    }
}

impl PartialEq<str> for ModelId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ModelId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderId;

    #[test]
    fn split_known_model_prefix_accepts_provider_ids() {
        assert_eq!(
            ProviderId::split_known_model_prefix("openrouter/openai/gpt-4o"),
            Some(("openrouter", "openai/gpt-4o"))
        );
    }

    #[test]
    fn split_known_model_prefix_rejects_model_namespaces() {
        assert_eq!(
            ProviderId::split_known_model_prefix("meta-llama/Llama-3.3-70B"),
            None
        );
    }
}
