// auth_store.rs — Credential store at ~/.mangocode/auth.json with optional
// MangoCode vault (`vault.enc`) as the preferred source when the vault exists
// and this process has an unlocked passphrase.
//
// Stores API keys and OAuth tokens for providers so users don't have to rely
// solely on environment variables.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use crate::oauth::OAuthTokens;
use crate::vault::{get_vault_passphrase, Vault};
use crate::ProviderId;

/// Vault entry keys that are not merged into [`AuthStore`] (infrastructure secrets).
const VAULT_SKIP_MERGE_KEYS: &[&str] = &["gateway", "tesseract", "ocr-tesseract"];

#[inline]
fn is_infrastructure_vault_key(pid: &str) -> bool {
    VAULT_SKIP_MERGE_KEYS.contains(&pid)
        || pid.starts_with("pipedream-")
        || pid.starts_with("pipedream_")
        || pid.starts_with("PIPEDREAM_")
}

/// Canonical provider id for in-memory and `auth.json` rows (matches [`AuthStore::api_key_for`]).
#[inline]
fn credential_storage_key(provider_id: &str) -> &str {
    if provider_id == "codex" {
        ProviderId::OPENAI_CODEX
    } else {
        provider_id
    }
}

/// Map a vault entry key to the auth-store key (`codex` → `openai-codex`).
#[inline]
fn vault_provider_key_to_storage_key(pid: &str) -> &str {
    credential_storage_key(pid)
}

/// Collapse legacy `codex` rows into `openai-codex` after loading JSON.
fn normalize_codex_alias_keys(store: &mut AuthStore) {
    if let Some(cred) = store.credentials.remove("codex") {
        store
            .credentials
            .entry(ProviderId::OPENAI_CODEX.to_string())
            .or_insert(cred);
    }
}

/// A stored credential for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StoredCredential {
    #[serde(rename = "api")]
    ApiKey { key: String },
    #[serde(rename = "oauth")]
    OAuthToken {
        access: String,
        refresh: String,
        expires: u64,
    },
}

/// Persistent credential store backed by `~/.mangocode/auth.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AuthStore {
    pub credentials: HashMap<String, StoredCredential>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthStoreClearReport {
    pub stored_credentials_removed: usize,
    pub vault_credentials_removed: usize,
    pub vault_locked: bool,
}

impl AuthStore {
    /// Merge Claude Max bearer credentials from `oauth_tokens.json` into `auth.json`
    /// after refresh or when the token file is ahead of the auth store.
    ///
    /// This keeps [`mangocode_api::providers::AnthropicMaxProvider`] and the
    /// provider registry aligned with the canonical OAuth token file.
    pub fn sync_anthropic_max_from_oauth_tokens(tokens: &OAuthTokens) -> anyhow::Result<()> {
        if !tokens.uses_bearer_auth() || tokens.access_token.is_empty() {
            return Ok(());
        }
        let mut store = Self::load();
        store.set_result(
            ProviderId::ANTHROPIC_MAX,
            StoredCredential::OAuthToken {
                access: tokens.access_token.clone(),
                refresh: tokens.refresh_token.clone().unwrap_or_default(),
                expires: tokens.expires_at_ms.map(|ms| ms as u64).unwrap_or(0),
            },
        )
    }

    /// Async version of sync_anthropic_max_from_oauth_tokens to avoid blocking I/O.
    pub async fn sync_anthropic_max_from_oauth_tokens_async(
        tokens: &OAuthTokens,
    ) -> anyhow::Result<()> {
        if !tokens.uses_bearer_auth() || tokens.access_token.is_empty() {
            return Ok(());
        }
        let mut store = Self::load_async().await;
        let cred = StoredCredential::OAuthToken {
            access: tokens.access_token.clone(),
            refresh: tokens.refresh_token.clone().unwrap_or_default(),
            expires: tokens.expires_at_ms.map(|ms| ms as u64).unwrap_or(0),
        };
        store
            .set_async_result(ProviderId::ANTHROPIC_MAX, cred)
            .await
    }

    /// Path to the auth store file.
    pub fn path() -> PathBuf {
        let dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".mangocode");
        dir.join("auth.json")
    }

    fn load_from_path(path: &Path) -> anyhow::Result<Self> {
        let data = match std::fs::read_to_string(path) {
            Ok(data) => data,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(err) => {
                return Err(err).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
    }

    async fn load_from_path_async(path: &Path) -> anyhow::Result<Self> {
        let data = match tokio::fs::read_to_string(path).await {
            Ok(data) => data,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(err) => {
                return Err(err).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
    }

    /// Load the store from disk (returns default if missing; logs and returns
    /// default if invalid).
    ///
    /// When `~/.mangocode/vault.enc` exists and the vault passphrase is cached for
    /// this session, credentials from the vault override the same provider keys
    /// in `auth.json` (vault first, JSON fallback).
    pub fn load() -> Self {
        let path = Self::path();
        let mut store = match Self::load_from_path(&path) {
            Ok(store) => store,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "failed to load auth store; using defaults"
                );
                Self::default()
            }
        };
        normalize_codex_alias_keys(&mut store);
        Self::merge_vault_over_json(&mut store);
        normalize_codex_alias_keys(&mut store);
        store
    }

    /// Async version of load() to avoid blocking the thread.
    pub async fn load_async() -> Self {
        let path = Self::path();
        let mut store = match Self::load_from_path_async(&path).await {
            Ok(store) => store,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "failed to load auth store; using defaults"
                );
                Self::default()
            }
        };
        normalize_codex_alias_keys(&mut store);
        Self::merge_vault_over_json(&mut store);
        normalize_codex_alias_keys(&mut store);
        store
    }

    /// Decode a vault secret string into a [`StoredCredential`].
    ///
    /// New writes use JSON-serialized [`StoredCredential`] (supports OAuth).
    /// Legacy `/vault set` entries are a raw API key string.
    fn credential_from_vault_secret(secret: &str) -> Option<StoredCredential> {
        let t = secret.trim();
        if t.is_empty() {
            return None;
        }
        if let Ok(c) = serde_json::from_str::<StoredCredential>(t) {
            return Some(c);
        }
        Some(StoredCredential::ApiKey { key: t.to_string() })
    }

    fn merge_vault_over_json(store: &mut AuthStore) {
        let vault = Vault::new();
        if !vault.exists() {
            return;
        }
        let Some(pass) = get_vault_passphrase() else {
            return;
        };
        let data = match vault.load(&pass) {
            Ok(d) => d,
            Err(_) => return,
        };
        for (pid, entry) in data.entries.iter() {
            if is_infrastructure_vault_key(pid.as_str()) {
                continue;
            }
            if let Some(cred) = Self::credential_from_vault_secret(&entry.secret) {
                let storage_key = vault_provider_key_to_storage_key(pid.as_str());
                store.credentials.insert(storage_key.to_string(), cred);
            }
        }
    }

    /// When the vault file exists and the session has an unlocked passphrase,
    /// persist this credential into the vault (encrypted). Always paired with
    /// `auth.json` writes elsewhere so JSON remains the offline fallback.
    fn mirror_credential_to_vault_if_unlocked(provider_id: &str, cred: &StoredCredential) {
        let vault = Vault::new();
        if !vault.exists() {
            return;
        }
        let Some(pass) = get_vault_passphrase() else {
            return;
        };
        let Ok(secret) = serde_json::to_string(cred) else {
            return;
        };
        let vault_key = credential_storage_key(provider_id);
        if let Err(err) = vault.set_secret(vault_key, &secret, &pass, None) {
            tracing::warn!(
                error = %err,
                provider_id = %vault_key,
                "failed to mirror credential into vault"
            );
        }
    }

    fn save_to_path(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    async fn save_to_path_async(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let json = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    fn clear_json_credentials(&mut self) -> usize {
        let removed = self.credentials.len();
        self.credentials.clear();
        removed
    }

    fn clear_user_vault_entries(data: &mut crate::vault::VaultData) -> usize {
        let keys: Vec<String> = data
            .entries
            .keys()
            .filter(|provider| !is_infrastructure_vault_key(provider.as_str()))
            .cloned()
            .collect();
        let removed = keys.len();
        for key in keys {
            data.entries.remove(&key);
        }
        if removed > 0 {
            data.version += 1;
        }
        removed
    }

    fn clear_unlocked_vault_credentials() -> anyhow::Result<Option<usize>> {
        let vault = Vault::new();
        if !vault.exists() {
            return Ok(Some(0));
        }
        let Some(passphrase) = get_vault_passphrase() else {
            return Ok(None);
        };
        let mut data = vault.load(&passphrase)?;
        let removed = Self::clear_user_vault_entries(&mut data);
        if removed > 0 {
            vault.save(&data, &passphrase)?;
        }
        Ok(Some(removed))
    }

    /// Persist the store to disk and report I/O/serialization errors.
    pub fn save_result(&self) -> anyhow::Result<()> {
        self.save_to_path(&Self::path())
    }

    /// Persist the store to disk (best-effort).
    pub fn save(&self) {
        if let Err(err) = self.save_result() {
            tracing::warn!(error = %err, "failed to save auth store");
        }
    }

    /// Async version of save() to avoid blocking the thread.
    pub async fn save_async_result(&self) -> anyhow::Result<()> {
        self.save_to_path_async(&Self::path()).await
    }

    /// Async version of save() to avoid blocking the thread.
    pub async fn save_async(&self) {
        if let Err(err) = self.save_async_result().await {
            tracing::warn!(error = %err, "failed to save auth store");
        }
    }

    /// Clear all user provider credentials from `auth.json` and, when the vault
    /// is unlocked, from `vault.enc` while preserving infrastructure vault rows.
    pub fn clear_all_credentials(&mut self) -> anyhow::Result<AuthStoreClearReport> {
        let stored_credentials_removed = self.clear_json_credentials();
        self.save_result()?;
        let mut report = AuthStoreClearReport {
            stored_credentials_removed,
            ..Default::default()
        };
        match Self::clear_unlocked_vault_credentials()? {
            Some(removed) => report.vault_credentials_removed = removed,
            None => report.vault_locked = true,
        }
        Ok(report)
    }

    /// Async version of clear_all_credentials() to avoid blocking auth.json I/O.
    pub async fn clear_all_credentials_async(&mut self) -> anyhow::Result<AuthStoreClearReport> {
        let stored_credentials_removed = self.clear_json_credentials();
        self.save_async_result().await?;
        let mut report = AuthStoreClearReport {
            stored_credentials_removed,
            ..Default::default()
        };
        match Self::clear_unlocked_vault_credentials()? {
            Some(removed) => report.vault_credentials_removed = removed,
            None => report.vault_locked = true,
        }
        Ok(report)
    }

    /// Store a credential for the given provider (persists immediately).
    ///
    /// Always writes `~/.mangocode/auth.json`. If the vault exists and this process
    /// has an unlocked passphrase, also stores an encrypted copy in the vault
    /// (vault preferred on next [`load`](Self::load) when unlocked).
    pub fn set_result(&mut self, provider_id: &str, cred: StoredCredential) -> anyhow::Result<()> {
        let key = credential_storage_key(provider_id).to_string();
        let previous = self.credentials.insert(key.clone(), cred.clone());
        if let Err(err) = self.save_result() {
            match previous {
                Some(previous) => {
                    self.credentials.insert(key, previous);
                }
                None => {
                    self.credentials.remove(&key);
                }
            }
            return Err(err);
        }
        Self::mirror_credential_to_vault_if_unlocked(&key, &cred);
        Ok(())
    }

    /// Async version of set_result() to avoid blocking auth.json I/O.
    pub async fn set_async_result(
        &mut self,
        provider_id: &str,
        cred: StoredCredential,
    ) -> anyhow::Result<()> {
        let key = credential_storage_key(provider_id).to_string();
        let previous = self.credentials.insert(key.clone(), cred.clone());
        if let Err(err) = self.save_async_result().await {
            match previous {
                Some(previous) => {
                    self.credentials.insert(key, previous);
                }
                None => {
                    self.credentials.remove(&key);
                }
            }
            return Err(err);
        }
        Self::mirror_credential_to_vault_if_unlocked(&key, &cred);
        Ok(())
    }

    /// Store a credential for the given provider (persists best-effort).
    pub fn set(&mut self, provider_id: &str, cred: StoredCredential) {
        if let Err(err) = self.set_result(provider_id, cred) {
            tracing::warn!(
                error = %err,
                provider_id = %credential_storage_key(provider_id),
                "failed to set auth credential"
            );
        }
    }

    /// Get the stored credential for a provider.
    pub fn get(&self, provider_id: &str) -> Option<&StoredCredential> {
        self.credentials.get(credential_storage_key(provider_id))
    }

    /// Remove the credential for a provider (persists immediately).
    ///
    /// Drops the entry from `auth.json` and, when the vault is unlocked, removes
    /// the matching provider entry from the vault.
    pub fn remove_result(&mut self, provider_id: &str) -> anyhow::Result<()> {
        let key = credential_storage_key(provider_id);
        let previous = self.credentials.remove(key);
        if let Err(err) = self.save_result() {
            if let Some(previous) = previous {
                self.credentials.insert(key.to_string(), previous);
            }
            return Err(err);
        }
        if let Some(pass) = get_vault_passphrase() {
            if Vault::new().exists() {
                let v = Vault::new();
                v.remove_secret(key, &pass)?;
                // Legacy vault rows may use the short alias.
                if key == ProviderId::OPENAI_CODEX {
                    v.remove_secret("codex", &pass)?;
                }
            }
        }
        Ok(())
    }

    /// Async version of remove_result() to avoid blocking auth.json I/O.
    pub async fn remove_async_result(&mut self, provider_id: &str) -> anyhow::Result<()> {
        let key = credential_storage_key(provider_id);
        let previous = self.credentials.remove(key);
        if let Err(err) = self.save_async_result().await {
            if let Some(previous) = previous {
                self.credentials.insert(key.to_string(), previous);
            }
            return Err(err);
        }
        if let Some(pass) = get_vault_passphrase() {
            if Vault::new().exists() {
                let v = Vault::new();
                v.remove_secret(key, &pass)?;
                // Legacy vault rows may use the short alias.
                if key == ProviderId::OPENAI_CODEX {
                    v.remove_secret("codex", &pass)?;
                }
            }
        }
        Ok(())
    }

    /// Remove the credential for a provider (persists best-effort).
    pub fn remove(&mut self, provider_id: &str) {
        if let Err(err) = self.remove_result(provider_id) {
            tracing::warn!(
                error = %err,
                provider_id = %credential_storage_key(provider_id),
                "failed to remove auth credential"
            );
        }
    }

    /// Get the API key for a provider, checking stored credentials first then
    /// falling back to the relevant environment variable.
    pub fn api_key_for(&self, provider_id: &str) -> Option<String> {
        let storage_key = credential_storage_key(provider_id);

        // Check stored credentials first
        if let Some(stored) = self.get(storage_key) {
            match stored {
                StoredCredential::ApiKey { key } => {
                    if !key.is_empty() {
                        return Some(key.clone());
                    }
                }
                StoredCredential::OAuthToken {
                    access, refresh, ..
                } if provider_id == "github-copilot" => {
                    if !refresh.is_empty() {
                        return Some(refresh.clone());
                    }
                    if !access.is_empty() {
                        return Some(access.clone());
                    }
                }
                StoredCredential::OAuthToken { access, .. } if provider_id == "anthropic-max" => {
                    if !access.is_empty() {
                        return Some(access.clone());
                    }
                }
                StoredCredential::OAuthToken {
                    access,
                    refresh,
                    expires,
                } if storage_key == ProviderId::OPENAI_CODEX => {
                    let expires_at_secs = if *expires > 0 {
                        Some(*expires / 1000)
                    } else {
                        None
                    };
                    if crate::oauth_config::codex_auth_is_usable(
                        access,
                        Some(refresh.as_str()),
                        expires_at_secs,
                    ) {
                        return Some(access.clone());
                    }
                }
                _ => {}
            }
        }

        // Vertex token fallback (when auth_mode=AccessToken).
        if provider_id == "google-vertex" {
            return std::env::var("VERTEX_ACCESS_TOKEN")
                .ok()
                .filter(|k| !k.is_empty());
        }

        // Fall back to environment variable
        let env_var = match provider_id {
            "anthropic" => "ANTHROPIC_API_KEY",
            "openai" => "OPENAI_API_KEY",
            "google" => "GOOGLE_API_KEY",
            "groq" => "GROQ_API_KEY",
            "cerebras" => "CEREBRAS_API_KEY",
            "deepseek" => "DEEPSEEK_API_KEY",
            "mistral" => "MISTRAL_API_KEY",
            "xai" => "XAI_API_KEY",
            "openrouter" => "OPENROUTER_API_KEY",
            "togetherai" | "together-ai" => "TOGETHER_API_KEY",
            "perplexity" => "PERPLEXITY_API_KEY",
            "cohere" => "COHERE_API_KEY",
            "deepinfra" => "DEEPINFRA_API_KEY",
            "venice" => "VENICE_API_KEY",
            "github-copilot" => "GITHUB_TOKEN",
            "azure" => "AZURE_API_KEY",
            "huggingface" => {
                return std::env::var("HF_TOKEN")
                    .or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
                    .or_else(|_| std::env::var("HUGGINGFACE_API_KEY"))
                    .ok()
                    .filter(|k| !k.is_empty());
            }
            "nvidia" => "NVIDIA_API_KEY",
            _ => return None,
        };
        std::env::var(env_var).ok().filter(|k| !k.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_for_codex_alias_reads_openai_codex_oauth() {
        let mut store = AuthStore::default();
        store.credentials.insert(
            ProviderId::OPENAI_CODEX.to_string(),
            StoredCredential::OAuthToken {
                access: "oauth-access-test".into(),
                refresh: "".into(),
                expires: 0,
            },
        );
        assert_eq!(
            store.api_key_for("codex").as_deref(),
            Some("oauth-access-test")
        );
        assert_eq!(
            store.api_key_for(ProviderId::OPENAI_CODEX).as_deref(),
            Some("oauth-access-test")
        );
        assert!(store.get("codex").is_some());
        assert!(std::ptr::eq(
            store.get("codex").unwrap() as *const StoredCredential,
            store.get(ProviderId::OPENAI_CODEX).unwrap() as *const StoredCredential
        ));
    }

    #[test]
    fn api_key_for_codex_rejects_expired_token_without_refresh() {
        let mut store = AuthStore::default();
        let past_ms = chrono::Utc::now().timestamp_millis().saturating_sub(60_000) as u64;
        store.credentials.insert(
            ProviderId::OPENAI_CODEX.to_string(),
            StoredCredential::OAuthToken {
                access: "oauth-access-test".into(),
                refresh: "".into(),
                expires: past_ms,
            },
        );
        assert_eq!(store.api_key_for("codex"), None);
        assert_eq!(store.api_key_for(ProviderId::OPENAI_CODEX), None);
    }

    #[test]
    fn api_key_for_codex_accepts_expired_token_with_refresh() {
        let mut store = AuthStore::default();
        let past_ms = chrono::Utc::now().timestamp_millis().saturating_sub(60_000) as u64;
        store.credentials.insert(
            ProviderId::OPENAI_CODEX.to_string(),
            StoredCredential::OAuthToken {
                access: "oauth-access-test".into(),
                refresh: "refresh-token".into(),
                expires: past_ms,
            },
        );
        assert_eq!(
            store.api_key_for("codex").as_deref(),
            Some("oauth-access-test")
        );
        assert_eq!(
            store.api_key_for(ProviderId::OPENAI_CODEX).as_deref(),
            Some("oauth-access-test")
        );
    }

    #[test]
    fn credential_storage_key_aliases_codex_to_canonical_id() {
        assert_eq!(
            super::credential_storage_key("codex"),
            ProviderId::OPENAI_CODEX
        );
        assert_eq!(
            super::credential_storage_key("openai-codex"),
            ProviderId::OPENAI_CODEX
        );
        assert_eq!(
            super::vault_provider_key_to_storage_key("codex"),
            ProviderId::OPENAI_CODEX
        );
    }

    #[test]
    fn infrastructure_vault_keys_skip_pipedream_rows() {
        assert!(super::is_infrastructure_vault_key("gateway"));
        assert!(super::is_infrastructure_vault_key("pipedream-client-id"));
        assert!(super::is_infrastructure_vault_key(
            "pipedream_client_secret"
        ));
        assert!(super::is_infrastructure_vault_key("PIPEDREAM_PROJECT_ID"));
        assert!(!super::is_infrastructure_vault_key("openai"));
    }

    fn temp_auth_store_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!(
                "mangocode-auth-store-test-{}",
                uuid::Uuid::new_v4()
            ))
            .join("auth.json")
    }

    #[test]
    fn load_from_path_defaults_when_missing() {
        let path = temp_auth_store_path();

        let store = AuthStore::load_from_path(&path).unwrap();

        assert!(store.credentials.is_empty());
    }

    #[test]
    fn load_from_path_rejects_malformed_auth_json() {
        let path = temp_auth_store_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{not-json").unwrap();

        let err = AuthStore::load_from_path(&path).unwrap_err();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());

        assert!(err.to_string().contains("failed to parse"));
    }

    #[tokio::test]
    async fn load_from_path_async_rejects_malformed_auth_json() {
        let path = temp_auth_store_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{not-json").unwrap();

        let err = AuthStore::load_from_path_async(&path).await.unwrap_err();
        let _ = std::fs::remove_dir_all(path.parent().unwrap());

        assert!(err.to_string().contains("failed to parse"));
    }

    #[test]
    fn clear_json_credentials_reports_removed_entries() {
        let mut store = AuthStore::default();
        store.credentials.insert(
            ProviderId::OPENAI.to_string(),
            StoredCredential::ApiKey {
                key: "sk-openai".to_string(),
            },
        );
        store.credentials.insert(
            ProviderId::OPENAI_CODEX.to_string(),
            StoredCredential::OAuthToken {
                access: "access".to_string(),
                refresh: "refresh".to_string(),
                expires: 0,
            },
        );

        assert_eq!(store.clear_json_credentials(), 2);
        assert!(store.credentials.is_empty());
    }

    #[test]
    fn clear_user_vault_entries_preserves_infrastructure_rows() {
        let mut data = crate::vault::VaultData::default();
        data.entries.insert(
            ProviderId::OPENAI.to_string(),
            crate::vault::VaultEntry {
                provider: ProviderId::OPENAI.to_string(),
                secret: "sk-openai".to_string(),
                label: None,
                updated_at: "2026-05-25T00:00:00Z".to_string(),
            },
        );
        data.entries.insert(
            "pipedream-client-id".to_string(),
            crate::vault::VaultEntry {
                provider: "pipedream-client-id".to_string(),
                secret: "client-id".to_string(),
                label: None,
                updated_at: "2026-05-25T00:00:00Z".to_string(),
            },
        );
        data.entries.insert(
            "gateway".to_string(),
            crate::vault::VaultEntry {
                provider: "gateway".to_string(),
                secret: "gateway-token".to_string(),
                label: None,
                updated_at: "2026-05-25T00:00:00Z".to_string(),
            },
        );

        assert_eq!(AuthStore::clear_user_vault_entries(&mut data), 1);
        assert!(!data.entries.contains_key(ProviderId::OPENAI));
        assert!(data.entries.contains_key("pipedream-client-id"));
        assert!(data.entries.contains_key("gateway"));
        assert_eq!(data.version, 1);
    }
}
