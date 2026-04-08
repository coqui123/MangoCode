use anyhow::{anyhow, Context};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, warn};

static FLAGS: OnceLock<RwLock<FeatureFlags>> = OnceLock::new();

const FLAGS_FILENAME: &str = "flags.json";
const REMOTE_CACHE_FILENAME: &str = "feature_flags.json";
const ENV_PREFIX: &str = "MANGOCODE_FLAG_";

// Predefined flag names
pub const FLAG_PROACTIVE_AGENT: &str = "proactive";
pub const FLAG_CRITIC_PERMISSIONS: &str = "critic_permissions";
pub const FLAG_LLM_COMPACTION: &str = "llm_compaction";
pub const FLAG_PROMPT_CACHING: &str = "prompt_caching";
pub const FLAG_AUTO_LSP: &str = "auto_lsp";
pub const FLAG_HIERARCHICAL_MEMORY: &str = "hierarchical_memory";

#[derive(Debug, Clone, Default)]
pub struct FeatureFlags {
    flags: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlag {
    pub id: String,
    pub key: String,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFlags {
    flags: HashMap<String, FeatureFlag>,
    fetched_at: u64,
}

#[derive(Debug, Deserialize)]
struct GrowthBookApiResponse {
    pub features: Vec<FeatureFlag>,
}

impl FeatureFlags {
    /// Load flags from ~/.mangocode/flags.json, then override with env vars.
    /// Env var format: MANGOCODE_FLAG_<NAME>=1|0
    pub fn init() -> Self {
        let flags = Self::merge_layers(None, Self::load_from_disk(), Self::env_overrides());
        Self { flags }
    }

    pub fn ensure_initialized() -> &'static RwLock<FeatureFlags> {
        FLAGS.get_or_init(|| RwLock::new(Self::init()))
    }

    pub fn is_enabled(name: &str) -> bool {
        Self::ensure_initialized()
            .read()
            .flags
            .get(name)
            .copied()
            .unwrap_or(false)
    }

    pub fn set(name: &str, value: bool) {
        let lock = Self::ensure_initialized();
        {
            let mut state = lock.write();
            state.flags.insert(name.to_string(), value);
            let _ = state.save_to_disk();
        }
    }

    pub fn refresh(&mut self) {
        self.flags = Self::merge_layers(None, Self::load_from_disk(), Self::env_overrides());
    }

    pub fn refresh_with_remote(&mut self, remote: HashMap<String, bool>) {
        self.flags =
            Self::merge_layers(Some(remote), Self::load_from_disk(), Self::env_overrides());
    }

    pub fn list_all() -> Vec<(String, bool)> {
        let state = Self::ensure_initialized().read();
        let mut names: BTreeSet<String> = Self::predefined_names()
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        names.extend(state.flags.keys().cloned());
        names
            .into_iter()
            .map(|name| {
                let enabled = state.flags.get(&name).copied().unwrap_or(false);
                (name, enabled)
            })
            .collect()
    }

    fn predefined_names() -> &'static [&'static str] {
        &[
            FLAG_PROACTIVE_AGENT,
            FLAG_CRITIC_PERMISSIONS,
            FLAG_LLM_COMPACTION,
            FLAG_PROMPT_CACHING,
            FLAG_AUTO_LSP,
            FLAG_HIERARCHICAL_MEMORY,
            "cached_microcompact",
        ]
    }

    fn default_flags() -> HashMap<String, bool> {
        let mut defaults = HashMap::new();
        defaults.insert("cached_microcompact".to_string(), true);
        defaults.insert(FLAG_LLM_COMPACTION.to_string(), true);
        defaults
    }

    fn load_from_disk() -> HashMap<String, bool> {
        let Some(path) = Self::flags_path() else {
            return HashMap::new();
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return HashMap::new();
        };
        serde_json::from_str::<HashMap<String, bool>>(&text).unwrap_or_default()
    }

    fn save_to_disk(&self) -> anyhow::Result<()> {
        let Some(path) = Self::flags_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.flags)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn env_overrides() -> HashMap<String, bool> {
        let mut overrides = HashMap::new();
        for (key, value) in std::env::vars() {
            let Some(raw_name) = key.strip_prefix(ENV_PREFIX) else {
                continue;
            };
            let Some(parsed) = parse_env_bool(&value) else {
                continue;
            };
            let name = raw_name.to_ascii_lowercase();
            overrides.insert(name, parsed);
        }
        overrides
    }

    fn merge_layers(
        remote: Option<HashMap<String, bool>>,
        local: HashMap<String, bool>,
        env: HashMap<String, bool>,
    ) -> HashMap<String, bool> {
        let mut merged = Self::default_flags();
        if let Some(remote) = remote {
            merged.extend(remote);
        }
        merged.extend(local);
        merged.extend(env);
        merged
    }

    fn flags_path() -> Option<PathBuf> {
        Some(dirs::home_dir()?.join(".mangocode").join(FLAGS_FILENAME))
    }
}

fn parse_env_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Backward-compatible wrapper while callers migrate to FeatureFlags.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureFlagManager;

impl FeatureFlagManager {
    pub fn new() -> Self {
        let _ = FeatureFlags::ensure_initialized();
        Self
    }

    pub fn flag(&self, name: &str) -> bool {
        FeatureFlags::is_enabled(name)
    }

    pub async fn fetch_flags_async(&self) -> anyhow::Result<()> {
        let remote = self.fetch_remote_flags_with_cache().await?;
        let lock = FeatureFlags::ensure_initialized();
        let mut state = lock.write();
        state.refresh_with_remote(remote);
        Ok(())
    }

    async fn fetch_remote_flags_with_cache(&self) -> anyhow::Result<HashMap<String, bool>> {
        if let Ok(cached) = self.load_cached_flags().await {
            if self.is_cache_valid(&cached) {
                debug!("Using cached remote feature flags");
                return Ok(Self::bool_map_from_cached(&cached));
            }
        }

        match self.fetch_from_api().await {
            Ok(cached) => {
                if let Err(e) = self.save_cached_flags(&cached).await {
                    warn!("Failed to save remote feature flag cache: {}", e);
                }
                Ok(Self::bool_map_from_cached(&cached))
            }
            Err(e) => {
                warn!("Failed to fetch remote feature flags: {}", e);
                if let Ok(cached) = self.load_cached_flags().await {
                    debug!("Using stale remote feature flag cache");
                    return Ok(Self::bool_map_from_cached(&cached));
                }
                warn!("No remote feature flag cache available; using local flags only");
                Ok(HashMap::new())
            }
        }
    }

    async fn fetch_from_api(&self) -> anyhow::Result<CachedFlags> {
        let api_endpoint = "https://api.growthbook.io/api/features";
        let api_key = std::env::var("GROWTHBOOK_API_KEY").ok();

        let mut builder = reqwest::Client::new().get(api_endpoint);
        if let Some(key) = api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = builder
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("Failed to fetch from GrowthBook API")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!(
                "GrowthBook API returned status {}: {}",
                status.as_u16(),
                response.text().await.unwrap_or_default()
            ));
        }

        let body = response
            .json::<GrowthBookApiResponse>()
            .await
            .context("Failed to parse GrowthBook API response")?;

        Ok(CachedFlags {
            flags: body
                .features
                .into_iter()
                .map(|f| (f.key.clone(), f))
                .collect(),
            fetched_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    fn is_cache_valid(&self, cached: &CachedFlags) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(cached.fetched_at) < 3600
    }

    async fn load_cached_flags(&self) -> anyhow::Result<CachedFlags> {
        let path = Self::remote_cache_path();
        let data = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let cached: CachedFlags =
            serde_json::from_str(&data).context("Failed to parse remote flag cache")?;
        Ok(cached)
    }

    async fn save_cached_flags(&self, cached: &CachedFlags) -> anyhow::Result<()> {
        let path = Self::remote_cache_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        let json = serde_json::to_string(cached).context("Failed to serialize remote flag cache")?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("Failed to write {}", path.display()))?;
        Ok(())
    }

    fn bool_map_from_cached(cached: &CachedFlags) -> HashMap<String, bool> {
        cached
            .flags
            .iter()
            .map(|(k, v)| (k.clone(), v.enabled))
            .collect()
    }

    fn remote_cache_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mangocode")
            .join(REMOTE_CACHE_FILENAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_bool_parser() {
        assert_eq!(parse_env_bool("1"), Some(true));
        assert_eq!(parse_env_bool("true"), Some(true));
        assert_eq!(parse_env_bool("0"), Some(false));
        assert_eq!(parse_env_bool("off"), Some(false));
        assert_eq!(parse_env_bool("maybe"), None);
    }

    #[test]
    fn test_list_all_includes_predefined() {
        let names: Vec<String> = FeatureFlags::list_all().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n == FLAG_PROACTIVE_AGENT));
        assert!(names.iter().any(|n| n == FLAG_LLM_COMPACTION));
    }

    #[test]
    fn test_merge_layers_precedence_remote_local_env() {
        let remote = HashMap::from([
            ("proactive".to_string(), false),
            ("remote_only".to_string(), true),
        ]);
        let local = HashMap::from([
            ("proactive".to_string(), true),
            ("local_only".to_string(), true),
        ]);
        let env = HashMap::from([
            ("proactive".to_string(), false),
            ("env_only".to_string(), true),
        ]);

        let merged = FeatureFlags::merge_layers(Some(remote), local, env);

        assert_eq!(merged.get("remote_only"), Some(&true));
        assert_eq!(merged.get("local_only"), Some(&true));
        assert_eq!(merged.get("env_only"), Some(&true));
        assert_eq!(merged.get("proactive"), Some(&false));
    }
}
