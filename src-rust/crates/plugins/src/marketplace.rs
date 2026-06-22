//! Plugin marketplace.
//!
//! Provides search, install, update, list, and uninstall for plugins
//! from the Claude registry.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A plugin entry from the marketplace registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceEntry {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub download_url: String,
    pub hash: String,
    pub tags: Vec<String>,
    pub updated_at: Option<u64>,
}

/// An installed plugin summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledPlugin {
    pub name: String,
    pub version: String,
    pub install_path: std::path::PathBuf,
    pub description: String,
}

// ---------------------------------------------------------------------------
// Marketplace API client
// ---------------------------------------------------------------------------

const REGISTRY_URL: &str = "https://registry.claude.ai/plugins";

async fn response_error_body(response: reqwest::Response, context: &str) -> String {
    response
        .text()
        .await
        .unwrap_or_else(|err| format!("<failed to read {context} error response body: {err}>"))
}

fn format_http_error(prefix: &str, status: reqwest::StatusCode, body: String) -> String {
    let body = body.trim();
    if body.is_empty() {
        format!("{prefix} {status}")
    } else {
        format!("{prefix} {status}: {body}")
    }
}

/// Search the marketplace for plugins matching `query`, optionally filtered by `tags`.
///
/// When `tags` is non-empty, `tags[]=tag` query parameters are appended to the URL.
pub async fn marketplace_search_filtered(
    query: &str,
    tags: &[&str],
) -> Result<Vec<MarketplaceEntry>, String> {
    let mut params: Vec<String> = Vec::new();

    if !query.is_empty() {
        params.push(format!("q={}", urlencoding::encode(query)));
    }
    for tag in tags {
        params.push(format!("tags[]={}", urlencoding::encode(tag)));
    }

    let url = if params.is_empty() {
        REGISTRY_URL.to_string()
    } else {
        format!("{}?{}", REGISTRY_URL, params.join("&"))
    };

    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = response_error_body(resp, "plugin marketplace registry").await;
        return Err(format_http_error("Registry returned", status, body));
    }

    resp.json::<Vec<MarketplaceEntry>>()
        .await
        .map_err(|e| format!("Parse error: {e}"))
}

/// Search the marketplace for plugins matching `query`.
///
/// Convenience wrapper around [`marketplace_search_filtered`] with no tag filter.
pub async fn marketplace_search(query: &str) -> Result<Vec<MarketplaceEntry>, String> {
    marketplace_search_filtered(query, &[]).await
}

/// Check all installed plugins for updates.
///
/// Returns `(name, current_version, latest_version)` for each plugin that has
/// a newer version available in the marketplace.
pub async fn marketplace_check_updates_all() -> Vec<(String, String, String)> {
    let installed = list_installed();
    let mut futures_vec = Vec::new();
    for plugin in &installed {
        let name = plugin.name.clone();
        let current = plugin.version.clone();
        futures_vec.push(async move {
            match marketplace_search(&name).await {
                Ok(results) => {
                    if let Some(latest) = results.iter().find(|e| e.name == name) {
                        if latest.version != current {
                            return Some((name, current, latest.version.clone()));
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        plugin = %name,
                        error = %err,
                        "failed to check plugin marketplace update"
                    );
                }
            }
            None
        });
    }
    futures::future::join_all(futures_vec)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// Install a plugin by name from the marketplace, or from a URL directly.
pub async fn marketplace_install(name_or_url: &str) -> Result<InstalledPlugin, String> {
    let entry = if name_or_url.starts_with("http") {
        // Direct URL install
        MarketplaceEntry {
            name: name_or_url
                .split('/')
                .next_back()
                .unwrap_or("plugin")
                .trim_end_matches(".zip")
                .to_string(),
            version: "0.0.0".to_string(),
            description: String::new(),
            author: String::new(),
            download_url: name_or_url.to_string(),
            hash: String::new(),
            tags: Vec::new(),
            updated_at: None,
        }
    } else {
        // Search by name
        let results = marketplace_search(name_or_url).await?;
        results
            .into_iter()
            .find(|e| e.name == name_or_url)
            .ok_or_else(|| format!("Plugin '{}' not found in marketplace", name_or_url))?
    };

    let install_dir = plugin_install_dir(&entry.name);
    std::fs::create_dir_all(&install_dir).map_err(|e| format!("Create dir: {e}"))?;

    // Download archive
    let resp = reqwest::get(&entry.download_url)
        .await
        .map_err(|e| format!("Download error: {e}"))?;
    let bytes = resp.bytes().await.map_err(|e| format!("Read bytes: {e}"))?;

    // Verify hash if provided
    if !entry.hash.is_empty() {
        use sha2::{Digest, Sha256};
        let computed = hex::encode(Sha256::digest(&bytes));
        if computed != entry.hash {
            return Err(format!(
                "Hash mismatch: expected {}, got {}",
                entry.hash, computed
            ));
        }
    }

    // Write to disk (assume .zip or direct .yaml file)
    let archive_path = install_dir.join("plugin.zip");
    std::fs::write(&archive_path, &bytes).map_err(|e| format!("Write: {e}"))?;

    // Try to unzip; if not a zip, treat as manifest YAML
    if try_unzip(&archive_path, &install_dir).is_err() {
        // Not a zip — assume it's the manifest directly
        let manifest_path = install_dir.join("manifest.yaml");
        std::fs::copy(&archive_path, &manifest_path).map_err(|e| format!("Copy: {e}"))?;
        cleanup_download_archive(&archive_path);
    } else {
        cleanup_download_archive(&archive_path);
    }

    Ok(InstalledPlugin {
        name: entry.name.clone(),
        version: entry.version.clone(),
        install_path: install_dir,
        description: entry.description.clone(),
    })
}

/// Try to unzip `archive` into `dest`. Returns Err if not a valid zip.
fn try_unzip(archive: &std::path::Path, dest: &std::path::Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    zip.extract(dest).map_err(|e| e.to_string())
}

fn cleanup_download_archive(path: &std::path::Path) {
    if let Err(err) = std::fs::remove_file(path) {
        if err.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "failed to remove plugin marketplace download archive"
            );
        }
    }
}

/// Check for an update to `name` and download if newer.
pub async fn marketplace_update(name: &str) -> Result<Option<String>, String> {
    let installed = list_installed();
    let current = installed
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| format!("Plugin '{}' is not installed", name))?;

    let results = marketplace_search(name).await?;
    let latest = results
        .iter()
        .find(|e| e.name == name)
        .ok_or_else(|| format!("Plugin '{}' not found in marketplace", name))?;

    if latest.version == current.version {
        return Ok(None); // Already up to date
    }

    marketplace_install(name).await?;
    Ok(Some(latest.version.clone()))
}

/// List all installed plugins.
pub fn list_installed() -> Vec<InstalledPlugin> {
    let plugins_dir = dirs::home_dir()
        .map(|h| h.join(".mangocode").join("plugins"))
        .unwrap_or_default();

    if !plugins_dir.exists() {
        return Vec::new();
    }

    let entries = match std::fs::read_dir(&plugins_dir) {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(
                path = %plugins_dir.display(),
                error = %err,
                "failed to list installed plugins"
            );
            return Vec::new();
        }
    };

    let mut installed = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!(
                    path = %plugins_dir.display(),
                    error = %err,
                    "failed to read installed plugin directory entry"
                );
                continue;
            }
        };
        let path = entry.path();
        let is_dir = match entry.file_type() {
            Ok(file_type) => file_type.is_dir(),
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to inspect installed plugin path"
                );
                continue;
            }
        };
        if !is_dir {
            continue;
        }
        let Some(name) = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
        else {
            continue;
        };

        let yaml_path = path.join("manifest.yaml");
        let json_path = path.join("manifest.json");

        let (version, description) = if yaml_path.exists() {
            let content = read_installed_manifest_or_empty(&yaml_path);
            let version =
                extract_yaml_str(&content, "version").unwrap_or_else(|| "0.0.0".to_string());
            let description = extract_yaml_str(&content, "description").unwrap_or_default();
            (version, description)
        } else if json_path.exists() {
            let content = read_installed_manifest_or_empty(&json_path);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                (
                    v["version"].as_str().unwrap_or("0.0.0").to_string(),
                    v["description"].as_str().unwrap_or("").to_string(),
                )
            } else {
                tracing::warn!(
                    path = %json_path.display(),
                    "failed to parse installed plugin manifest JSON"
                );
                ("0.0.0".to_string(), String::new())
            }
        } else {
            ("0.0.0".to_string(), String::new())
        };

        installed.push(InstalledPlugin {
            name,
            version,
            install_path: path,
            description,
        });
    }

    installed
}

fn read_installed_manifest_or_empty(path: &std::path::Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| {
        tracing::warn!(
            path = %path.display(),
            error = %err,
            "failed to read installed plugin manifest"
        );
        String::new()
    })
}

/// Uninstall a plugin by removing its directory.
pub fn marketplace_uninstall(name: &str) -> Result<(), String> {
    let dir = plugin_install_dir(name);
    if !dir.exists() {
        return Err(format!("Plugin '{}' is not installed", name));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("Remove dir: {e}"))
}

fn plugin_install_dir(name: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".mangocode")
        .join("plugins")
        .join(name)
}

fn extract_yaml_str(content: &str, key: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix(&format!("{key}:")) {
            return Some(rest.trim().trim_matches('"').trim_matches('\'').to_string());
        }
    }
    None
}
