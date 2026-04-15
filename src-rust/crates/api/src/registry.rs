// registry.rs — Registry of all available LLM providers.
//
// Holds an `Arc<dyn LlmProvider>` for each registered provider and exposes
// lookup, health-check, and default-provider helpers.

use std::collections::HashMap;
use std::sync::Arc;

use mangocode_core::ProviderId;

use crate::client::ClientConfig;
use crate::provider::LlmProvider;
use crate::provider_types::ProviderStatus;
use crate::providers::{
    AnthropicMaxProvider, AnthropicProvider, AzureProvider, BedrockProvider, CohereProvider,
    CopilotProvider, GoogleProvider, MinimaxProvider, OpenAiProvider, VertexOpenAiProvider,
};

fn vault_key_aliases(provider_id: &str) -> &'static [&'static str] {
    match provider_id {
        // Historical / UI aliases
        "together-ai" => &["togetherai"],
        "togetherai" => &["together-ai"],
        "qwen" => &["alibaba"],
        "alibaba" => &["qwen"],
        "moonshotai" => &["moonshot"],
        "zhipuai" => &["zhipu"],
        "vultr" => &["vultr-ai"],
        "vultr-ai" => &["vultr"],
        _ => &[],
    }
}

fn env_or_vault(env_var: &str, provider_id: &str) -> Option<String> {
    std::env::var(env_var)
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            let vault = mangocode_core::Vault::new();
            mangocode_core::get_vault_passphrase().and_then(|passphrase| {
                // Canonical provider id first, then known aliases.
                vault.get_secret(provider_id, &passphrase)
                    .ok()
                    .flatten()
                    .or_else(|| {
                        for alias in vault_key_aliases(provider_id) {
                            if let Ok(Some(v)) = vault.get_secret(alias, &passphrase) {
                                return Some(v);
                            }
                        }
                        None
                    })
            })
        })
}

/// Registry of all available LLM providers.
/// Holds `Arc<dyn LlmProvider>` for each registered provider.
pub struct ProviderRegistry {
    providers: HashMap<ProviderId, Arc<dyn LlmProvider>>,
    default_provider_id: ProviderId,
}

impl ProviderRegistry {
    /// Create an empty registry with Anthropic as the default provider ID.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider_id: ProviderId::new(ProviderId::ANTHROPIC),
        }
    }

    /// Register a provider. Returns `&mut self` for builder chaining.
    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) -> &mut Self {
        let id = provider.id().clone();
        self.providers.insert(id, provider);
        self
    }

    /// Set the default provider by ID.
    ///
    /// # Panics
    /// Panics if no provider with that ID has been registered.
    pub fn set_default(&mut self, id: ProviderId) -> &mut Self {
        assert!(
            self.providers.contains_key(&id),
            "set_default: provider '{}' is not registered",
            id,
        );
        self.default_provider_id = id;
        self
    }

    /// Get a provider by ID.
    pub fn get(&self, id: &ProviderId) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(id)
    }

    /// Get the default provider.
    pub fn default_provider(&self) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(&self.default_provider_id)
    }

    /// Get the default provider ID.
    pub fn default_provider_id(&self) -> &ProviderId {
        &self.default_provider_id
    }

    /// List all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<&ProviderId> {
        self.providers.keys().collect()
    }

    /// Check health of all providers sequentially.
    /// Returns `(provider_id, status)` pairs.
    pub async fn check_all_health(&self) -> Vec<(ProviderId, ProviderStatus)> {
        let mut results = Vec::new();
        for (id, provider) in &self.providers {
            let status = provider
                .health_check()
                .await
                .unwrap_or(ProviderStatus::Unavailable {
                    reason: "health check failed".to_string(),
                });
            results.push((id.clone(), status));
        }
        results
    }

    /// Convenience: build a registry with just Anthropic registered as the
    /// default provider.  Takes the same [`ClientConfig`] that
    /// [`AnthropicClient`] takes.
    ///
    /// [`AnthropicClient`]: crate::client::AnthropicClient
    pub fn with_anthropic(config: ClientConfig) -> Self {
        let mut registry = Self::new();
        let provider = Arc::new(AnthropicProvider::from_config(config));
        registry.register(provider);
        registry
    }

    /// Register [`GoogleProvider`] if `GOOGLE_API_KEY` or
    /// `GOOGLE_GENERATIVE_AI_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_google_if_key_set(&mut self) -> &mut Self {
        let key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_GENERATIVE_AI_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| env_or_vault("GOOGLE_API_KEY", "google"))
            .or_else(|| env_or_vault("GOOGLE_GENERATIVE_AI_API_KEY", "google"));
        if let Some(key) = key {
            let provider = Arc::new(GoogleProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`OpenAiProvider`] if `OPENAI_API_KEY` is set in the
    /// environment or vault. Returns `&mut self` for builder chaining.
    pub fn with_openai_if_key_set(&mut self) -> &mut Self {
        if let Some(key) = env_or_vault("OPENAI_API_KEY", "openai") {
            let provider = Arc::new(OpenAiProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`AzureProvider`] if `AZURE_API_KEY` and `AZURE_RESOURCE_NAME`
    /// are set in the environment.  Returns `&mut self` for builder chaining.
    pub fn with_azure_if_configured(&mut self) -> &mut Self {
        if let Some(p) = AzureProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`BedrockProvider`] if AWS credentials are available in the
    /// `AWS_BEARER_TOKEN_BEDROCK`).  Returns `&mut self` for builder chaining.
    pub fn with_bedrock_if_configured(&mut self) -> &mut Self {
        if let Some(p) = BedrockProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CopilotProvider`] if `GITHUB_TOKEN` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_copilot_if_configured(&mut self) -> &mut Self {
        if let Some(p) = CopilotProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CohereProvider`] if `COHERE_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_cohere_if_key_set(&mut self) -> &mut Self {
        if let Some(p) = CohereProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`VertexOpenAiProvider`] if `VERTEX_PROJECT_ID` is set in the
    /// environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_vertex_if_configured(&mut self) -> &mut Self {
        if let Some(p) = VertexOpenAiProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`AnthropicMaxProvider`] if an `anthropic-max` OAuth token is
    /// stored in the auth store. Returns `&mut self` for builder chaining.
    pub fn with_anthropic_max_if_configured(&mut self) -> &mut Self {
        if let Some(p) = AnthropicMaxProvider::from_auth_store() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Build a registry with **all** providers that have credentials configured
    /// in the environment.  Anthropic is always the default provider.
    ///
    /// This is the recommended constructor for production use.
    pub fn from_environment(anthropic_config: ClientConfig) -> Self {
        let mut registry = Self::with_anthropic(anthropic_config);
        registry
            .with_openai_if_key_set()
            .with_google_if_key_set()
            .with_azure_if_configured()
            .with_bedrock_if_configured()
            .with_copilot_if_configured()
            .with_cohere_if_key_set()
            .with_vertex_if_configured()
            .with_anthropic_max_if_configured()
            .with_available_providers();
        registry
    }

    /// Build a registry that checks **both** environment variables and the
    /// persistent [`AuthStore`] (`~/.mangocode/auth.json`) for credentials.
    ///
    /// This ensures that API keys stored via `/connect` or `mangocode auth` are
    /// picked up at startup, not just env vars.  Falls back to
    /// `from_environment` for providers that only support env-var config, and
    /// adds any extra providers that have keys in the auth store.
    ///
    /// [`AuthStore`]: mangocode_core::AuthStore
    pub fn from_environment_with_auth_store(anthropic_config: ClientConfig) -> Self {
        // Start with env-based registration.
        let mut registry = Self::from_environment(anthropic_config);

        // Now check the auth store for providers that weren't registered from
        // env vars.
        let auth_store = mangocode_core::AuthStore::load();

        for provider_id in auth_store.credentials.keys() {
            let pid = mangocode_core::ProviderId::new(provider_id.as_str());
            // Skip if already registered from env vars.
            if registry.get(&pid).is_some() {
                continue;
            }
            // Try to get a usable key from the auth store.
            if let Some(key) = auth_store.api_key_for(provider_id) {
                if key.is_empty() {
                    continue;
                }
                use crate::providers::openai_compat_providers as p;
                let provider: Option<Arc<dyn LlmProvider>> = match provider_id.as_str() {
                    "anthropic-max" => {
                        // Claude Max uses Bearer auth — create from the OAuth access token
                        Some(Arc::new(AnthropicMaxProvider::new(key)))
                    }
                    "openai" => Some(Arc::new(OpenAiProvider::new(key))),
                    "google" => Some(Arc::new(GoogleProvider::new(key))),
                    "minimax" => Some(Arc::new(MinimaxProvider::new(key))),
                    "github-copilot" => Some(Arc::new(CopilotProvider::new(key))),
                    "cohere" => Some(Arc::new(CohereProvider::new(key))),
                    "groq" => Some(Arc::new(p::groq().with_api_key(key))),
                    "mistral" => Some(Arc::new(p::mistral().with_api_key(key))),
                    "deepseek" => Some(Arc::new(p::deepseek().with_api_key(key))),
                    "xai" => Some(Arc::new(p::xai().with_api_key(key))),
                    "openrouter" => Some(Arc::new(p::openrouter().with_api_key(key))),
                    "togetherai" | "together-ai" => {
                        Some(Arc::new(p::together_ai().with_api_key(key)))
                    }
                    "perplexity" => Some(Arc::new(p::perplexity().with_api_key(key))),
                    "cerebras" => Some(Arc::new(p::cerebras().with_api_key(key))),
                    "deepinfra" => Some(Arc::new(p::deepinfra().with_api_key(key))),
                    "venice" => Some(Arc::new(p::venice().with_api_key(key))),
                    "huggingface" => Some(Arc::new(p::huggingface().with_api_key(key))),
                    "nvidia" => Some(Arc::new(p::nvidia().with_api_key(key))),
                    "siliconflow" => Some(Arc::new(p::siliconflow().with_api_key(key))),
                    "sambanova" => Some(Arc::new(p::sambanova().with_api_key(key))),
                    "moonshot" => Some(Arc::new(p::moonshot().with_api_key(key))),
                    "zhipu" => Some(Arc::new(p::zhipu().with_api_key(key))),
                    "qwen" => Some(Arc::new(p::qwen().with_api_key(key))),
                    "nebius" => Some(Arc::new(p::nebius().with_api_key(key))),
                    "novita" => Some(Arc::new(p::novita().with_api_key(key))),
                    "ovhcloud" => Some(Arc::new(p::ovhcloud().with_api_key(key))),
                    "scaleway" => Some(Arc::new(p::scaleway().with_api_key(key))),
                    "vultr" | "vultr-ai" => Some(Arc::new(p::vultr_ai().with_api_key(key))),
                    "baseten" => Some(Arc::new(p::baseten().with_api_key(key))),
                    "friendli" => Some(Arc::new(p::friendli().with_api_key(key))),
                    "upstage" => Some(Arc::new(p::upstage().with_api_key(key))),
                    "stepfun" => Some(Arc::new(p::stepfun().with_api_key(key))),
                    "fireworks" => Some(Arc::new(p::fireworks().with_api_key(key))),
                    // Vertex: the stored "key" is an access token; project/location
                    // still come from VERTEX_* env vars.
                    "google-vertex" => {
                        use crate::providers::vertex_openai::{VertexAuthMode, VertexConfig};
                        let project_id = std::env::var("VERTEX_PROJECT_ID").unwrap_or_default();
                        if !project_id.is_empty() {
                            let location = std::env::var("VERTEX_LOCATION")
                                .unwrap_or_else(|_| "us-central1".to_string());
                            let model = std::env::var("VERTEX_MODEL")
                                .unwrap_or_else(|_| "google/gemini-2.5-flash".to_string());
                            let cfg = VertexConfig {
                                project_id,
                                location,
                                model,
                                auth_mode: VertexAuthMode::AccessToken,
                                access_token: Some(key),
                                base_url_override: None,
                            };
                            Some(Arc::new(VertexOpenAiProvider::new(cfg)))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(p) = provider {
                    registry.register(p);
                }
            }
        }

        registry
    }

    /// Register all providers that have environment variable credentials set.
    ///
    /// Local providers (Ollama, LM Studio, llama.cpp) are always registered
    /// regardless of credentials — `health_check()` will report them as
    /// unavailable if the server is not running.
    ///
    /// Remote API-key providers are only registered when their respective
    /// environment variables are set (non-empty).
    ///
    /// Returns `&mut self` for builder chaining.
    pub fn with_available_providers(&mut self) -> &mut Self {
        use crate::providers::openai_compat_providers as p;

        fn register_if_key_set<F>(registry: &mut ProviderRegistry, env_var: &str, vault_key: &str, f: F)
        where
            F: FnOnce(String) -> Arc<dyn LlmProvider>,
        {
            if let Some(key) = env_or_vault(env_var, vault_key) {
                registry.register(f(key));
            }
        }

        // Local providers — always try to register.
        self.register(Arc::new(p::ollama()));
        self.register(Arc::new(p::lm_studio()));
        self.register(Arc::new(p::llama_cpp()));

        // Remote providers — only register when an API key is present.
        register_if_key_set(self, "DEEPSEEK_API_KEY", "deepseek", |key| {
            Arc::new(p::deepseek().with_api_key(key))
        });
        register_if_key_set(self, "GROQ_API_KEY", "groq", |key| Arc::new(p::groq().with_api_key(key)));
        register_if_key_set(self, "XAI_API_KEY", "xai", |key| Arc::new(p::xai().with_api_key(key)));
        register_if_key_set(self, "OPENROUTER_API_KEY", "openrouter", |key| {
            Arc::new(p::openrouter().with_api_key(key))
        });
        register_if_key_set(self, "TOGETHER_API_KEY", "together-ai", |key| {
            Arc::new(p::together_ai().with_api_key(key))
        });
        register_if_key_set(self, "PERPLEXITY_API_KEY", "perplexity", |key| {
            Arc::new(p::perplexity().with_api_key(key))
        });
        register_if_key_set(self, "CEREBRAS_API_KEY", "cerebras", |key| {
            Arc::new(p::cerebras().with_api_key(key))
        });
        register_if_key_set(self, "DEEPINFRA_API_KEY", "deepinfra", |key| {
            Arc::new(p::deepinfra().with_api_key(key))
        });
        register_if_key_set(self, "VENICE_API_KEY", "venice", |key| Arc::new(p::venice().with_api_key(key)));
        register_if_key_set(self, "DASHSCOPE_API_KEY", "qwen", |key| Arc::new(p::qwen().with_api_key(key)));
        register_if_key_set(self, "MISTRAL_API_KEY", "mistral", |key| {
            Arc::new(p::mistral().with_api_key(key))
        });
        register_if_key_set(self, "SAMBANOVA_API_KEY", "sambanova", |key| {
            Arc::new(p::sambanova().with_api_key(key))
        });
        register_if_key_set(self, "HF_TOKEN", "huggingface", |key| {
            Arc::new(p::huggingface().with_api_key(key))
        });
        register_if_key_set(self, "MINIMAX_API_KEY", "minimax", |key| Arc::new(MinimaxProvider::new(key)));
        register_if_key_set(self, "NVIDIA_API_KEY", "nvidia", |key| Arc::new(p::nvidia().with_api_key(key)));
        register_if_key_set(self, "SILICONFLOW_API_KEY", "siliconflow", |key| {
            Arc::new(p::siliconflow().with_api_key(key))
        });
        register_if_key_set(self, "MOONSHOT_API_KEY", "moonshotai", |key| {
            Arc::new(p::moonshot().with_api_key(key))
        });
        register_if_key_set(self, "ZHIPU_API_KEY", "zhipuai", |key| Arc::new(p::zhipu().with_api_key(key)));
        register_if_key_set(self, "NEBIUS_API_KEY", "nebius", |key| Arc::new(p::nebius().with_api_key(key)));
        register_if_key_set(self, "NOVITA_API_KEY", "novita", |key| Arc::new(p::novita().with_api_key(key)));
        register_if_key_set(self, "OVHCLOUD_API_KEY", "ovhcloud", |key| {
            Arc::new(p::ovhcloud().with_api_key(key))
        });
        register_if_key_set(self, "SCALEWAY_API_KEY", "scaleway", |key| {
            Arc::new(p::scaleway().with_api_key(key))
        });
        register_if_key_set(self, "VULTR_API_KEY", "vultr", |key| Arc::new(p::vultr_ai().with_api_key(key)));
        register_if_key_set(self, "BASETEN_API_KEY", "baseten", |key| Arc::new(p::baseten().with_api_key(key)));
        register_if_key_set(self, "FRIENDLI_TOKEN", "friendli", |key| Arc::new(p::friendli().with_api_key(key)));
        register_if_key_set(self, "UPSTAGE_API_KEY", "upstage", |key| Arc::new(p::upstage().with_api_key(key)));
        register_if_key_set(self, "STEPFUN_API_KEY", "stepfun", |key| Arc::new(p::stepfun().with_api_key(key)));
        register_if_key_set(self, "FIREWORKS_API_KEY", "fireworks", |key| {
            Arc::new(p::fireworks().with_api_key(key))
        });
        self
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::vault_key_aliases;

    #[test]
    fn vault_key_aliases_cover_known_provider_id_variants() {
        assert_eq!(vault_key_aliases("together-ai"), &["togetherai"]);
        assert_eq!(vault_key_aliases("togetherai"), &["together-ai"]);
        assert_eq!(vault_key_aliases("qwen"), &["alibaba"]);
        assert_eq!(vault_key_aliases("alibaba"), &["qwen"]);
        assert_eq!(vault_key_aliases("moonshotai"), &["moonshot"]);
        assert_eq!(vault_key_aliases("zhipuai"), &["zhipu"]);
        assert_eq!(vault_key_aliases("vultr"), &["vultr-ai"]);
        assert_eq!(vault_key_aliases("vultr-ai"), &["vultr"]);
        assert!(vault_key_aliases("openai").is_empty());
    }
}
