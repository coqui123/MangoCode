// update_check.rs — Background update checker.
//
// Fetches the latest release from GitHub and compares it against the running
// version.  Results are cached on disk for 24 hours so we never hammer the
// GitHub API on every startup.

use std::time::Duration;

use semver::Version;

pub const GITHUB_RELEASES_API_URL: &str =
    "https://api.github.com/repos/coqui123/MangoCode/releases";
pub const GITHUB_RELEASES_API_LATEST_URL: &str =
    "https://api.github.com/repos/coqui123/MangoCode/releases/latest";
pub const GITHUB_RELEASES_PAGE: &str = "https://github.com/coqui123/MangoCode/releases";
pub const GITHUB_REPO_URL: &str = "https://github.com/coqui123/MangoCode";
const CHECK_INTERVAL_HOURS: u64 = 24;

/// Information about an available update.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub release_url: String,
    pub has_update: bool,
}

/// Check for a newer version of MangoCode in the background.
///
/// Returns `Some(UpdateInfo)` when a newer release exists on GitHub.
/// The result is cached for `CHECK_INTERVAL_HOURS` hours so repeated
/// calls within that window are served from disk without a network round-trip.
pub async fn check_for_updates() -> Option<UpdateInfo> {
    let current = env!("CARGO_PKG_VERSION").to_string();

    // --- 24-hour rate-limit cache -------------------------------------------
    if let Some(cache_path) = update_cache_path() {
        if cache_path.exists() {
            if let Ok(metadata) = std::fs::metadata(&cache_path) {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(elapsed) = modified.elapsed() {
                        if elapsed < Duration::from_secs(CHECK_INTERVAL_HOURS * 3600) {
                            // Cache is still fresh — use the stored version.
                            if let Ok(cached) = std::fs::read_to_string(&cache_path) {
                                let cached = cached.trim().to_string();
                                if cached.is_empty() {
                                    return None;
                                }
                                let has_update = is_newer(&cached, &current).unwrap_or(false);
                                if has_update {
                                    return Some(UpdateInfo {
                                        current_version: current,
                                        latest_version: cached.clone(),
                                        release_url: format!(
                                            "{}/tag/v{}",
                                            GITHUB_RELEASES_PAGE, cached
                                        ),
                                        has_update: true,
                                    });
                                }
                                return None;
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Network fetch -------------------------------------------------------
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .user_agent(format!("MangoCode/{}", current))
        .build()
        .ok()?;

    let resp = client
        .get(GITHUB_RELEASES_API_LATEST_URL)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }

    let json: serde_json::Value = resp.json().await.ok()?;
    let tag = json.get("tag_name").and_then(|v| v.as_str())?;
    let latest = tag.trim_start_matches('v').to_string();
    let html_url = json
        .get("html_url")
        .and_then(|v| v.as_str())
        .unwrap_or(GITHUB_RELEASES_PAGE)
        .to_string();

    // Cache the fetched version so we don't hit GitHub again for 24 h.
    if let Some(cache_path) = update_cache_path() {
        if let Some(parent) = cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache_path, &latest);
    }

    let has_update = is_newer(&latest, &current).unwrap_or(false);
    if has_update {
        Some(UpdateInfo {
            current_version: current,
            latest_version: latest,
            release_url: html_url,
            has_update: true,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn update_cache_path() -> Option<std::path::PathBuf> {
    dirs::cache_dir().map(|d| d.join("mangocode").join("update_check.txt"))
}

/// Compare two semver strings.
///
/// Returns `None` when either version is not valid SemVer.
pub fn is_newer(latest: &str, current: &str) -> Option<bool> {
    let latest = Version::parse(latest.trim().trim_start_matches('v')).ok()?;
    let current = Version::parse(current.trim().trim_start_matches('v')).ok()?;
    Some(latest > current)
}

#[cfg(test)]
mod tests {
    use super::is_newer;

    #[test]
    fn newer_minor() {
        assert_eq!(is_newer("0.1.0", "0.0.7"), Some(true));
    }

    #[test]
    fn same_version() {
        assert_eq!(is_newer("0.0.7", "0.0.7"), Some(false));
    }

    #[test]
    fn older_version() {
        assert_eq!(is_newer("0.0.5", "0.0.7"), Some(false));
    }

    #[test]
    fn major_bump() {
        assert_eq!(is_newer("1.0.0", "0.9.9"), Some(true));
    }

    #[test]
    fn leading_v_is_accepted() {
        assert_eq!(is_newer("v0.1.0", "v0.0.7"), Some(true));
    }

    #[test]
    fn prerelease_is_ordered_by_semver() {
        assert_eq!(is_newer("1.0.0", "1.0.0-rc.1"), Some(true));
        assert_eq!(is_newer("1.0.0-rc.1", "1.0.0"), Some(false));
    }

    #[test]
    fn malformed_versions_are_unknown() {
        assert_eq!(is_newer("latest", "0.0.7"), None);
        assert_eq!(is_newer("0.0.8", "dev"), None);
    }
}
