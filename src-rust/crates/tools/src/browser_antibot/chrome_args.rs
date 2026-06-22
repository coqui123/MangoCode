//! Chromium CLI flags broadly aligned with nodriver `nodriver/core/config.py` defaults.

#[cfg(unix)]
const DEV_SHM: &[&str] = &["--disable-dev-shm-usage"];

#[cfg(not(unix))]
const DEV_SHM: &[&str] = &[];

/// Extra flags merged into [`chromiumoxide::browser::BrowserConfig`] alongside defaults.
pub fn stealth_extra_browser_args(lang: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = vec![
        "--remote-allow-origins=*".into(),
        "--no-first-run".into(),
        "--no-service-autorun".into(),
        "--no-default-browser-check".into(),
        "--homepage=about:blank".into(),
        "--no-pings".into(),
        "--password-store=basic".into(),
        "--disable-infobars".into(),
        "--disable-breakpad".into(),
        "--disable-session-crashed-bubble".into(),
        "--disable-search-engine-choice-screen".into(),
        "--disable-features=IsolateOrigins,site-per-process".into(),
        "--disable-blink-features=AutomationControlled".into(),
        format!("--lang={}", lang.unwrap_or("en-US,en;q=0.9")),
        format!("--accept-lang={}", lang.unwrap_or("en-US,en;q=0.9")),
    ];
    for a in DEV_SHM {
        out.push((*a).to_string());
    }
    if super::browser_expert_env_enabled() {
        out.push("--disable-site-isolation-trials".into());
    }
    out
}
