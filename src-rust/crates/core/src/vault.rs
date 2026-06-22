//! Encrypted local credential vault.
//!
//! Stores API keys encrypted with AES-256-GCM, derived from a user passphrase
//! via Argon2id. Secrets are decrypted in-memory only when needed.

use aes_gcm::{aead::Aead, Aes256Gcm, KeyInit, Nonce};
use anyhow::Context;
use argon2::{Algorithm, Argon2, Params, Version};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rand::RngCore;
use reqwest::{header::HeaderValue, ClientBuilder, Proxy};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use zeroize::{Zeroize, Zeroizing};

const VAULT_FILENAME: &str = "vault.enc";
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const GATEWAY_VAULT_KEY: &str = "gateway";

fn mangocode_config_dir() -> anyhow::Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".mangocode"))
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory for MangoCode config"))
}

fn gateway_config_path() -> anyhow::Result<PathBuf> {
    Ok(mangocode_config_dir()?.join("gateway.json"))
}

fn pipedream_config_path() -> anyhow::Result<PathBuf> {
    Ok(mangocode_config_dir()?.join("pipedream.json"))
}

fn default_vault_path() -> PathBuf {
    dirs::home_dir()
        .map(|home| home.join(".mangocode"))
        .unwrap_or_else(|| PathBuf::from(".mangocode"))
        .join(VAULT_FILENAME)
}

fn create_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn load_optional_json_file<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    let data = match std::fs::read_to_string(path) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    serde_json::from_str(&data)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn set_private_file_perms_best_effort(_path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::metadata(_path) {
            Ok(metadata) => {
                let mut perms = metadata.permissions();
                perms.set_mode(0o600);
                if let Err(err) = std::fs::set_permissions(_path, perms) {
                    tracing::warn!(
                        path = %_path.display(),
                        error = %err,
                        "failed to set private file permissions"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    path = %_path.display(),
                    error = %err,
                    "failed to inspect file permissions"
                );
            }
        }
    }
    // Keep the argument intentionally "used" on non-Unix builds too.
    let _ = _path;
}

fn replace_file_best_effort(tmp: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    // On Unix, `rename` replaces the destination atomically.
    // On Windows, `std::fs::rename` fails if the destination exists, so we
    // remove then rename (best-effort atomicity).
    match std::fs::rename(tmp, dest) {
        Ok(()) => Ok(()),
        Err(e) => {
            #[cfg(windows)]
            {
                // If the destination already exists, try removing it and retry.
                if dest.exists() {
                    match std::fs::remove_file(dest) {
                        Ok(()) => return std::fs::rename(tmp, dest),
                        Err(remove_err) if remove_err.kind() == std::io::ErrorKind::NotFound => {
                            return std::fs::rename(tmp, dest);
                        }
                        Err(remove_err) => return Err(remove_err),
                    }
                }
            }
            Err(e)
        }
    }
}

/// A single stored credential.
#[derive(Serialize, Deserialize, Clone)]
pub struct VaultEntry {
    /// Provider identifier (e.g. "anthropic", "openai", "google", "azure")
    pub provider: String,
    /// The secret value (API key, access token, etc.)
    pub secret: String,
    /// Optional metadata
    pub label: Option<String>,
    /// When this entry was last updated (RFC 3339)
    pub updated_at: String,
}

/// The full vault contents (serialized to JSON, then encrypted).
#[derive(Serialize, Deserialize, Default)]
pub struct VaultData {
    pub entries: HashMap<String, VaultEntry>,
    pub version: u32,
}

/// Configuration for OneCLI-style gateway proxy mode.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GatewayConfig {
    /// Whether to route API calls through the gateway.
    pub enabled: bool,
    /// Gateway URL (e.g. "http://localhost:10255").
    pub url: String,
    /// Access token for the Proxy-Authorization header.
    ///
    /// For best security, prefer storing this token in the vault under
    /// provider key `"gateway"`; this field exists for backward compatibility
    /// and for setups that want to use gateway mode without a vault.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
}

impl GatewayConfig {
    /// Load from ~/.mangocode/gateway.json
    pub fn load() -> Option<Self> {
        let path = match gateway_config_path() {
            Ok(path) => path,
            Err(err) => {
                tracing::warn!(error = %err, "failed to locate gateway config");
                return None;
            }
        };
        match Self::load_from_path(&path) {
            Ok(config) => config,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "failed to load gateway config; using defaults"
                );
                None
            }
        }
    }

    /// Load from ~/.mangocode/gateway.json while preserving read/parse errors.
    pub fn load_result() -> anyhow::Result<Option<Self>> {
        Self::load_from_path(&gateway_config_path()?)
    }

    pub(crate) fn load_from_path(path: &Path) -> anyhow::Result<Option<Self>> {
        load_optional_json_file(path)
    }

    /// Save to ~/.mangocode/gateway.json
    pub fn save(&self) -> anyhow::Result<()> {
        let path = mangocode_config_dir()?.join("gateway.json");
        create_parent_dir(&path)?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        set_private_file_perms_best_effort(&path);
        Ok(())
    }
}

/// Optional fallback configuration for Pipedream MCP.
/// Stored in ~/.mangocode/pipedream.json (not encrypted, but with restricted file permissions).
/// Sensitive credentials should be stored in the encrypted vault instead.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PipedreamConfig {
    /// Legacy plaintext OAuth client ID fallback. New writes should use the vault instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Legacy plaintext OAuth client secret fallback. New writes should use the vault instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// Legacy plaintext Pipedream project ID fallback. New writes should use the vault instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Environment (e.g., "development", "production")
    #[serde(default = "default_environment")]
    pub environment: String,
    /// Optional account/workspace ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// Optional MCP server URL for self-hosted Pipedream instances.
    /// Defaults to "https://remote.mcp.pipedream.net/v3" when not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_url: Option<String>,
    /// Optional OAuth token URL for self-hosted Pipedream instances.
    /// Defaults to "https://api.pipedream.com/v1/oauth/token" when not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
}

fn default_environment() -> String {
    "development".to_string()
}

impl PipedreamConfig {
    /// Load from ~/.mangocode/pipedream.json
    pub fn load() -> Option<Self> {
        let path = match pipedream_config_path() {
            Ok(path) => path,
            Err(err) => {
                tracing::warn!(error = %err, "failed to locate Pipedream config");
                return None;
            }
        };
        match Self::load_from_path(&path) {
            Ok(config) => config,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "failed to load Pipedream config; using defaults"
                );
                None
            }
        }
    }

    /// Load from ~/.mangocode/pipedream.json while preserving read/parse errors.
    pub fn load_result() -> anyhow::Result<Option<Self>> {
        Self::load_from_path(&pipedream_config_path()?)
    }

    pub(crate) fn load_from_path(path: &Path) -> anyhow::Result<Option<Self>> {
        load_optional_json_file(path)
    }

    /// Save to ~/.mangocode/pipedream.json
    pub fn save(&self) -> anyhow::Result<()> {
        let path = mangocode_config_dir()?.join("pipedream.json");
        create_parent_dir(&path)?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        set_private_file_perms_best_effort(&path);
        Ok(())
    }

    /// Return the MCP server URL, falling back to the Pipedream-hosted default.
    pub fn mcp_url(&self) -> String {
        self.mcp_url
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("https://remote.mcp.pipedream.net/v3")
            .to_string()
    }

    /// Return the OAuth token URL, falling back to the Pipedream-hosted default.
    pub fn token_url(&self) -> String {
        self.token_url
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("https://api.pipedream.com/v1/oauth/token")
            .to_string()
    }
}

/// Manages the encrypted vault file.
pub struct Vault {
    path: PathBuf,
}

impl Default for Vault {
    fn default() -> Self {
        Self::new()
    }
}

impl Vault {
    pub fn new() -> Self {
        Self {
            path: default_vault_path(),
        }
    }

    fn derive_key(passphrase: &str, salt: &[u8]) -> anyhow::Result<[u8; 32]> {
        let mut key = [0u8; 32];
        // Pin Argon2id parameters explicitly rather than relying on
        // `Argon2::default()`. These are the OWASP-recommended minimums and match
        // the argon2 0.5.x defaults, so existing vaults derive an identical key —
        // but pinning them protects users from a silent crate-default change
        // making previously-saved vaults undecryptable on a future upgrade.
        // m=19456 KiB, t=2 iterations, p=1 lane, 32-byte output.
        let params = Params::new(19_456, 2, 1, Some(32))
            .map_err(|err| anyhow::anyhow!("invalid Argon2 parameters: {err}"))?;
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
            .hash_password_into(passphrase.as_bytes(), salt, &mut key)
            .map_err(|err| anyhow::anyhow!("key derivation failed: {err}"))?;
        Ok(key)
    }

    /// Encrypt vault data and write to disk.
    /// File format: [salt:16][nonce:12][ciphertext:...]
    pub fn save(&self, data: &VaultData, passphrase: &str) -> anyhow::Result<()> {
        let json = Zeroizing::new(serde_json::to_vec(data)?);

        let mut salt = [0u8; SALT_LEN];
        rand::thread_rng().fill_bytes(&mut salt);

        let mut key = Self::derive_key(passphrase, &salt)?;
        let cipher = Aes256Gcm::new_from_slice(&key)?;
        key.zeroize();

        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, json.as_ref())
            .map_err(|e| anyhow::anyhow!("encryption failed: {}", e))?;

        let mut file_data = Vec::with_capacity(SALT_LEN + NONCE_LEN + ciphertext.len());
        file_data.extend_from_slice(&salt);
        file_data.extend_from_slice(&nonce_bytes);
        file_data.extend_from_slice(&ciphertext);

        create_parent_dir(&self.path)?;
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, &file_data)?;
        replace_file_best_effort(&tmp, &self.path)?;
        set_private_file_perms_best_effort(&self.path);

        Ok(())
    }

    /// Read and decrypt vault data from disk.
    pub fn load(&self, passphrase: &str) -> anyhow::Result<VaultData> {
        let file_data = std::fs::read(&self.path)?;
        if file_data.len() < SALT_LEN + NONCE_LEN + 1 {
            anyhow::bail!("vault file too short");
        }

        let salt = &file_data[..SALT_LEN];
        let nonce_bytes = &file_data[SALT_LEN..SALT_LEN + NONCE_LEN];
        let ciphertext = &file_data[SALT_LEN + NONCE_LEN..];

        let mut key = Self::derive_key(passphrase, salt)?;
        let cipher = Aes256Gcm::new_from_slice(&key)?;
        key.zeroize();
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = Zeroizing::new(
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| anyhow::anyhow!("decryption failed — wrong passphrase?"))?,
        );

        let data: VaultData = serde_json::from_slice(&plaintext)?;
        Ok(data)
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn get_secret(&self, provider: &str, passphrase: &str) -> anyhow::Result<Option<String>> {
        if !self.exists() {
            return Ok(None);
        }
        let data = self.load(passphrase)?;
        Ok(data.entries.get(provider).map(|e| e.secret.clone()))
    }

    pub fn set_secret(
        &self,
        provider: &str,
        secret: &str,
        passphrase: &str,
        label: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut data = if self.exists() {
            self.load(passphrase)?
        } else {
            VaultData::default()
        };

        data.entries.insert(
            provider.to_string(),
            VaultEntry {
                provider: provider.to_string(),
                secret: secret.to_string(),
                label: label.map(|s| s.to_string()),
                updated_at: chrono::Utc::now().to_rfc3339(),
            },
        );
        data.version += 1;

        self.save(&data, passphrase)
    }

    pub fn remove_secret(&self, provider: &str, passphrase: &str) -> anyhow::Result<()> {
        if !self.exists() {
            return Ok(());
        }
        let mut data = self.load(passphrase)?;
        data.entries.remove(provider);
        data.version += 1;
        self.save(&data, passphrase)
    }

    pub fn list_providers(
        &self,
        passphrase: &str,
    ) -> anyhow::Result<Vec<(String, Option<String>, String)>> {
        if !self.exists() {
            return Ok(vec![]);
        }
        let data = self.load(passphrase)?;
        let mut rows: Vec<(String, Option<String>, String)> = data
            .entries
            .values()
            .map(|e| (e.provider.clone(), e.label.clone(), e.updated_at.clone()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(rows)
    }
}

static VAULT_PASSPHRASE: Lazy<RwLock<Option<Zeroizing<String>>>> = Lazy::new(|| RwLock::new(None));

/// Return the cached vault passphrase for this process.
pub fn get_vault_passphrase() -> Option<String> {
    VAULT_PASSPHRASE
        .read()
        .as_ref()
        .map(|p| p.as_str().to_string())
}

/// Set the vault passphrase for this session.
pub fn set_vault_passphrase(passphrase: String) {
    *VAULT_PASSPHRASE.write() = Some(Zeroizing::new(passphrase));
}

/// Clear the vault passphrase (e.g. on session end).
pub fn clear_vault_passphrase() {
    *VAULT_PASSPHRASE.write() = None;
}

fn gateway_access_token_from_vault() -> Option<String> {
    let vault = Vault::new();
    get_vault_passphrase().and_then(|passphrase| {
        vault
            .get_secret(GATEWAY_VAULT_KEY, &passphrase)
            .ok()
            .flatten()
            .filter(|t| !t.trim().is_empty())
    })
}

/// Build a reqwest client builder that applies gateway proxy settings when enabled.
pub fn reqwest_client_builder() -> ClientBuilder {
    let mut builder = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

    if let Some(gw) = GatewayConfig::load().filter(|g| g.enabled) {
        let token = gw
            .access_token
            .clone()
            .filter(|t| !t.trim().is_empty())
            .or_else(gateway_access_token_from_vault);
        if let Some(token) = token {
            if let Ok(auth) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                if let Ok(proxy) = Proxy::all(&gw.url).map(|p| p.custom_http_auth(auth)) {
                    builder = builder.proxy(proxy);
                }
            }
        }
    }

    builder
}

/// Resolve an API key for any provider.
/// Resolution order: config → env var → vault → None
pub fn resolve_api_key_for_provider(
    config_key: Option<&str>,
    env_var: &str,
    vault_provider: &str,
) -> Option<String> {
    config_key
        .map(|k| k.to_string())
        .filter(|k| !k.is_empty())
        .or_else(|| std::env::var(env_var).ok().filter(|k| !k.is_empty()))
        .or_else(|| {
            let vault = Vault::new();
            get_vault_passphrase()
                .and_then(|passphrase| vault.get_secret(vault_provider, &passphrase).ok().flatten())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn create_parent_dir_allows_current_directory_paths() {
        create_parent_dir(Path::new("vault.enc")).unwrap();
    }

    #[test]
    fn vault_roundtrip() {
        let dir = TempDir::new().unwrap();
        let vault = Vault {
            path: dir.path().join("vault.enc"),
        };
        let passphrase = "test-passphrase-123";

        vault
            .set_secret("anthropic", "sk-ant-xxxxx", passphrase, Some("main key"))
            .unwrap();
        vault
            .set_secret("openai", "sk-openai-yyyyy", passphrase, None)
            .unwrap();

        assert_eq!(
            vault.get_secret("anthropic", passphrase).unwrap(),
            Some("sk-ant-xxxxx".to_string())
        );
        assert_eq!(
            vault.get_secret("openai", passphrase).unwrap(),
            Some("sk-openai-yyyyy".to_string())
        );
        assert_eq!(vault.get_secret("nonexistent", passphrase).unwrap(), None);
        assert!(vault.get_secret("anthropic", "wrong-passphrase").is_err());

        vault.remove_secret("openai", passphrase).unwrap();
        assert_eq!(vault.get_secret("openai", passphrase).unwrap(), None);

        let providers = vault.list_providers(passphrase).unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].0, "anthropic");
    }

    #[cfg(windows)]
    #[test]
    fn vault_save_overwrites_existing_file_on_windows() {
        let dir = TempDir::new().unwrap();
        let vault = Vault {
            path: dir.path().join("vault.enc"),
        };
        let passphrase = "test-passphrase-123";

        // First write creates the file.
        vault
            .set_secret("openai", "sk-openai-1", passphrase, Some("first"))
            .unwrap();

        // Second write must overwrite the existing vault.enc. On Windows, a plain
        // `std::fs::rename(tmp, dest)` fails if dest exists, so this guards
        // against regressions.
        vault
            .set_secret("openai", "sk-openai-2", passphrase, Some("second"))
            .unwrap();

        assert_eq!(
            vault.get_secret("openai", passphrase).unwrap(),
            Some("sk-openai-2".to_string())
        );

        let providers = vault.list_providers(passphrase).unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].0, "openai");
    }

    #[test]
    fn gateway_config_roundtrip() {
        let config = GatewayConfig {
            enabled: true,
            url: "http://localhost:10255".to_string(),
            access_token: Some("test-token".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: GatewayConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.url, "http://localhost:10255");
        assert!(parsed.enabled);
    }

    #[test]
    fn gateway_config_load_from_path_rejects_malformed_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gateway.json");
        std::fs::write(&path, "{not-json").unwrap();

        let err = GatewayConfig::load_from_path(&path).unwrap_err();

        assert!(err.to_string().contains("failed to parse"));
    }

    #[test]
    fn pipedream_config_load_from_path_rejects_malformed_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pipedream.json");
        std::fs::write(&path, "{not-json").unwrap();

        let err = PipedreamConfig::load_from_path(&path).unwrap_err();

        assert!(err.to_string().contains("failed to parse"));
    }
}
