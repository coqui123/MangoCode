// settings_sync.rs — Settings Sync
//
// Syncs user settings and AGENTS.md memory files between a local MangoCode
// installation and claude.ai via:
//   - Upload (interactive CLI, fire-and-forget at startup)
//   - Download (CCR / MANGOCODE_REMOTE=1, blocking before plugin load)
//
// Authentication requires OAuth (Bearer token). API-key-only users are
// skipped silently.
//
// The sync API stores a flat key→value map where keys are canonical file paths
// and values are the UTF-8 file contents (JSON or Markdown).

use crate::Settings;
use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SYNC_TIMEOUT_SECS: u64 = 10;
const DEFAULT_MAX_RETRIES: u32 = 3;
/// 500 KB per-file size limit (matches backend enforcement).
const MAX_FILE_SIZE_BYTES: u64 = 500 * 1024;

// ---------------------------------------------------------------------------
// Sync key helpers
// ---------------------------------------------------------------------------

/// Canonical sync key for the global user settings file.
pub const SYNC_KEY_USER_SETTINGS: &str = "~/.mangocode/settings.json";
/// Canonical sync key for the global user memory file.
pub const SYNC_KEY_USER_MEMORY: &str = "~/.mangocode/AGENTS.md";

/// Canonical sync key for per-project settings (keyed by git-remote hash).
pub fn sync_key_project_settings(project_id: &str) -> String {
    format!("projects/{project_id}/.mangocode/settings.local.json")
}

/// Canonical sync key for per-project memory (keyed by git-remote hash).
pub fn sync_key_project_memory(project_id: &str) -> String {
    format!("projects/{project_id}/AGENTS.local.md")
}

// ---------------------------------------------------------------------------
// API wire types
// ---------------------------------------------------------------------------

/// Content field in the GET response — flat string key/value map.
#[derive(Debug, Deserialize)]
struct UserSyncContent {
    entries: HashMap<String, String>,
}

/// Full GET /api/claude_code/user_settings response.
#[derive(Debug, Deserialize)]
struct UserSyncData {
    #[serde(rename = "userId")]
    user_id: Option<String>,
    version: Option<u64>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    checksum: Option<String>,
    content: UserSyncContent,
}

/// PUT response (partial — only fields we care about).
#[derive(Debug, Deserialize, Default)]
struct UploadResponse {
    checksum: Option<String>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
}

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// Data returned by a successful download.
#[derive(Debug, Clone, Default)]
pub struct SyncedData {
    /// Parsed user settings JSON (if the `user_settings` key was present).
    pub settings: Option<Value>,
    /// Raw file contents keyed by their sync keys.
    pub memory_files: HashMap<String, String>,
    /// Remote version for cache invalidation.
    pub version: Option<u64>,
    /// Remote checksum for ETag-style caching.
    pub checksum: Option<String>,
}

// ---------------------------------------------------------------------------
// SettingsSyncManager
// ---------------------------------------------------------------------------

/// Manages uploading and downloading settings/memory files to/from claude.ai.
pub struct SettingsSyncManager {
    /// OAuth bearer token for authentication.
    pub oauth_token: String,
    /// Base API URL (default: https://api.anthropic.com).
    pub base_url: String,
    http: reqwest::Client,
}

impl SettingsSyncManager {
    /// Create a new manager.
    pub fn new(oauth_token: String, base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(SYNC_TIMEOUT_SECS))
            .build()
            .unwrap_or_default();
        Self {
            oauth_token,
            base_url,
            http,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/api/claude_code/user_settings", self.base_url)
    }

    fn auth_headers(&self) -> [(&'static str, String); 2] {
        [
            ("Authorization", format!("Bearer {}", self.oauth_token)),
            ("anthropic-beta", "oauth-2025-04-20".to_string()),
        ]
    }

    // -----------------------------------------------------------------------
    // Download
    // -----------------------------------------------------------------------

    /// Download remote settings and memory files.
    ///
    /// Returns `Ok(None)` when the server has no data for this user (404).
    /// Fails open — callers should treat errors as "no remote data".
    pub async fn download(&self, cached_checksum: Option<&str>) -> Result<Option<SyncedData>> {
        let mut req = self.http.get(self.endpoint());
        for (name, value) in self.auth_headers() {
            req = req.header(name, value);
        }
        if let Some(checksum) = cached_checksum {
            req = req.header("If-None-Match", format!("\"{}\"", checksum));
        }

        let resp = req.send().await?;

        let status = resp.status().as_u16();
        if status == 404 {
            debug!("Settings sync: no remote data (404)");
            return Ok(None);
        }
        if status == 304 {
            debug!("Settings sync: remote data unchanged (304)");
            return Ok(None);
        }
        if status != 200 {
            anyhow::bail!("Settings sync download: unexpected status {}", status);
        }

        let UserSyncData {
            user_id,
            version,
            last_modified,
            checksum,
            content,
        } = resp.json().await?;

        if let Some(ref uid) = user_id {
            debug!(user_id = %uid, "Settings sync: authenticated as user");
        }
        if let Some(ref lm) = last_modified {
            debug!(last_modified = %lm, "Settings sync: server timestamp");
        }

        Ok(Some(entries_to_synced_data(
            content.entries,
            version,
            checksum,
        )))
    }

    /// Download with exponential-backoff retry.
    async fn download_with_retry(
        &self,
        cached_checksum: Option<&str>,
    ) -> Result<Option<SyncedData>> {
        let mut last_err = anyhow::anyhow!("No attempts made");
        for attempt in 1..=(DEFAULT_MAX_RETRIES + 1) {
            match self.download(cached_checksum).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let msg = e.to_string();
                    // Auth failures are terminal
                    if msg.contains("401") || msg.contains("403") {
                        return Err(e);
                    }
                    warn!(
                        attempt,
                        max = DEFAULT_MAX_RETRIES,
                        error = %e,
                        "Settings sync download failed, will retry"
                    );
                    last_err = e;
                    if attempt <= DEFAULT_MAX_RETRIES {
                        tokio::time::sleep(retry_delay(attempt)).await;
                    }
                }
            }
        }
        Err(last_err)
    }

    /// Apply downloaded entries to local files.
    ///
    /// Writes settings and memory files to the appropriate local paths,
    /// enforcing the 500 KB per-file size limit.
    pub async fn apply_to_local(&self, data: &SyncedData, project_id: Option<&str>) -> ApplyResult {
        let mut result = ApplyResult::default();

        // Global user settings
        if let Some(ref settings_json) = data.settings {
            let path = Settings::config_dir().join("settings.json");
            let content = serde_json::to_string_pretty(settings_json).unwrap_or_default();
            match write_file_for_sync(&path, &content).await {
                Ok(()) => {
                    result.settings_written = true;
                    result.settings_applied = true;
                    result.applied_count += 1;
                }
                Err(e) => warn!("Settings sync: failed to write user settings: {}", e),
            }
        }

        // Global user memory
        if let Some(memory) = data.memory_files.get(SYNC_KEY_USER_MEMORY) {
            let path = Settings::config_dir().join("AGENTS.md");
            match write_file_for_sync(&path, memory).await {
                Ok(()) => {
                    result.memory_written = true;
                    result.applied_count += 1;
                }
                Err(e) => warn!("Settings sync: failed to write user memory: {}", e),
            }
        }

        // Project-specific files
        if let Some(pid) = project_id {
            let proj_settings_key = sync_key_project_settings(pid);
            if let Some(content) = data.memory_files.get(&proj_settings_key) {
                let path = std::env::current_dir()
                    .unwrap_or_default()
                    .join(".mangocode")
                    .join("settings.local.json");
                match write_file_for_sync(&path, content).await {
                    Ok(()) => {
                        result.settings_written = true;
                        result.applied_count += 1;
                    }
                    Err(e) => {
                        warn!("Settings sync: failed to write project settings: {}", e)
                    }
                }
            }

            let proj_memory_key = sync_key_project_memory(pid);
            if let Some(content) = data.memory_files.get(&proj_memory_key) {
                let path = std::env::current_dir()
                    .unwrap_or_default()
                    .join("AGENTS.local.md");
                match write_file_for_sync(&path, content).await {
                    Ok(()) => {
                        result.memory_written = true;
                        result.applied_count += 1;
                    }
                    Err(e) => {
                        warn!("Settings sync: failed to write project memory: {}", e)
                    }
                }
            }
        }

        result
    }

    // -----------------------------------------------------------------------
    // Upload
    // -----------------------------------------------------------------------

    /// Upload local settings and memory files to remote.
    ///
    /// Compares with existing remote entries and only uploads changed keys.
    pub async fn upload(&self, local_entries: HashMap<String, String>) -> Result<()> {
        // Fetch current remote state for diff
        let remote_entries = match self.download_with_retry(None).await? {
            Some(data) => data.memory_files,
            None => HashMap::new(),
        };

        // Only send keys that have changed
        let changed: HashMap<String, String> = local_entries
            .into_iter()
            .filter(|(k, v)| remote_entries.get(k).map(|rv| rv != v).unwrap_or(true))
            .collect();

        if changed.is_empty() {
            debug!("Settings sync: no changes to upload");
            return Ok(());
        }

        debug!(
            count = changed.len(),
            "Settings sync: uploading changed entries"
        );
        self.put_entries(changed).await
    }

    async fn put_entries(&self, entries: HashMap<String, String>) -> Result<()> {
        let body = serde_json::json!({ "entries": entries });
        let mut req = self
            .http
            .put(self.endpoint())
            .header("Content-Type", "application/json")
            .json(&body);
        for (name, value) in self.auth_headers() {
            req = req.header(name, value);
        }

        let resp = req.send().await?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            anyhow::bail!("Settings sync upload: unexpected status {}", status);
        }
        let resp_body: UploadResponse = resp.json().await.unwrap_or_default();
        debug!(
            checksum = ?resp_body.checksum,
            last_modified = ?resp_body.last_modified,
            "Settings sync: upload response"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Fire-and-forget background upload (called from startup)
    // -----------------------------------------------------------------------

    /// Spawn a fire-and-forget upload task. Errors are logged but not propagated.
    ///
    /// Call this right after auth is established.  The task will:
    ///   1. Read local settings and AGENTS.md files
    ///   2. Fetch current remote state for diffing
    ///   3. Upload only changed entries
    pub fn upload_in_background(token: String, base_url: String) {
        tokio::spawn(async move {
            let mgr = SettingsSyncManager::new(token, base_url);
            let entries = collect_local_entries(None).await;
            if let Err(e) = mgr.upload(entries).await {
                warn!("Settings sync: background upload failed: {}", e);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Apply result
// ---------------------------------------------------------------------------

/// Summary of what `apply_to_local` wrote.
#[derive(Debug, Default)]
pub struct ApplyResult {
    pub applied_count: usize,
    pub settings_applied: bool,
    pub settings_written: bool,
    pub memory_written: bool,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert raw sync entries into the `SyncedData` structure.
///
/// The user settings entry is parsed as JSON; memory files are kept as-is.
fn entries_to_synced_data(
    entries: HashMap<String, String>,
    version: Option<u64>,
    checksum: Option<String>,
) -> SyncedData {
    let mut data = SyncedData::default();

    for (key, value) in entries {
        if key == SYNC_KEY_USER_SETTINGS {
            data.settings = serde_json::from_str(&value).ok();
        } else {
            data.memory_files.insert(key, value);
        }
    }

    data.version = version;
    data.checksum = checksum;
    data
}

/// Collect local files that should be uploaded.
///
/// Reads global user settings and AGENTS.md, plus (if `project_id` is given)
/// project-local settings and AGENTS.local.md.  Files larger than 500 KB or
/// that cannot be read are silently omitted.
pub async fn collect_local_entries(project_id: Option<&str>) -> HashMap<String, String> {
    let mut entries = HashMap::new();

    // Global user settings
    let settings_path = Settings::config_dir().join("settings.json");
    if let Some(content) = try_read_for_sync(&settings_path).await {
        entries.insert(SYNC_KEY_USER_SETTINGS.to_string(), content);
    }

    // Global user memory
    let memory_path = Settings::config_dir().join("AGENTS.md");
    if let Some(content) = try_read_for_sync(&memory_path).await {
        entries.insert(SYNC_KEY_USER_MEMORY.to_string(), content);
    }

    // Project-specific files
    if let Some(pid) = project_id {
        let cwd = std::env::current_dir().unwrap_or_default();

        let local_settings = cwd.join(".mangocode").join("settings.local.json");
        if let Some(content) = try_read_for_sync(&local_settings).await {
            entries.insert(sync_key_project_settings(pid), content);
        }

        let local_memory = cwd.join("AGENTS.local.md");
        if let Some(content) = try_read_for_sync(&local_memory).await {
            entries.insert(sync_key_project_memory(pid), content);
        }
    }

    entries
}

/// Try to read a file, applying the 500 KB size limit.
/// Returns `None` if the file doesn't exist, is empty, or exceeds the limit.
async fn try_read_for_sync(path: &PathBuf) -> Option<String> {
    let meta = tokio::fs::metadata(path).await.ok()?;
    if meta.len() > MAX_FILE_SIZE_BYTES {
        debug!(path = %path.display(), "Settings sync: file exceeds 500 KB limit, skipping");
        return None;
    }
    let content = tokio::fs::read_to_string(path).await.ok()?;
    if content.trim().is_empty() {
        return None;
    }
    Some(content)
}

/// Write `content` to `path`, creating parent directories as needed.
async fn write_file_for_sync(path: &PathBuf, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, content).await?;
    Ok(())
}

/// Exponential backoff delay for retry attempt `n` (1-indexed), capped at 30 s.
fn retry_delay(attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(30);
    let secs: u64 = 1u64.checked_shl(shift).unwrap_or(u64::MAX).min(30);
    Duration::from_secs(secs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_sync_keys() {
        assert_eq!(SYNC_KEY_USER_SETTINGS, "~/.mangocode/settings.json");
        assert_eq!(SYNC_KEY_USER_MEMORY, "~/.mangocode/AGENTS.md");
        assert_eq!(
            sync_key_project_settings("abc123"),
            "projects/abc123/.mangocode/settings.local.json"
        );
        assert_eq!(
            sync_key_project_memory("abc123"),
            "projects/abc123/AGENTS.local.md"
        );
    }

    #[test]
    fn test_entries_to_synced_data_settings_parsed() {
        let mut entries = HashMap::new();
        entries.insert(
            SYNC_KEY_USER_SETTINGS.to_string(),
            r#"{"model":"claude-3"}"#.to_string(),
        );
        entries.insert(SYNC_KEY_USER_MEMORY.to_string(), "# My notes".to_string());

        let data = entries_to_synced_data(entries, Some(7), Some("sha256:abc".to_string()));
        assert!(data.settings.is_some());
        assert_eq!(data.settings.unwrap()["model"], json!("claude-3"));
        assert_eq!(
            data.memory_files.get(SYNC_KEY_USER_MEMORY).unwrap(),
            "# My notes"
        );
        assert_eq!(data.version, Some(7));
        assert_eq!(data.checksum.as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn test_entries_to_synced_data_invalid_json_settings() {
        let mut entries = HashMap::new();
        entries.insert(SYNC_KEY_USER_SETTINGS.to_string(), "not-json".to_string());
        let data = entries_to_synced_data(entries, None, None);
        // Malformed settings JSON → field is None (graceful degradation)
        assert!(data.settings.is_none());
    }

    #[test]
    fn test_entries_to_synced_data_empty() {
        let data = entries_to_synced_data(HashMap::new(), None, None);
        assert!(data.settings.is_none());
        assert!(data.memory_files.is_empty());
    }

    #[test]
    fn test_retry_delay_progression() {
        assert_eq!(retry_delay(1), Duration::from_secs(1));
        assert_eq!(retry_delay(2), Duration::from_secs(2));
        assert_eq!(retry_delay(3), Duration::from_secs(4));
        assert_eq!(retry_delay(4), Duration::from_secs(8));
        assert_eq!(retry_delay(5), Duration::from_secs(16));
        // Capped at 30 s
        assert_eq!(retry_delay(6), Duration::from_secs(30));
        assert_eq!(retry_delay(10), Duration::from_secs(30));
    }
}
