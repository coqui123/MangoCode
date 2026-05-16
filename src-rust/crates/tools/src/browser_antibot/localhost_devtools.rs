//! Localhost-only Chromium DevTools HTTP JSON helpers (parity with nodriver inspector HTTP probes).

pub async fn devtools_json_list(debug_port: u16) -> Result<serde_json::Value, anyhow::Error> {
    anyhow::ensure!(debug_port != 0, "debug_port required");
    let url = format!("http://127.0.0.1:{debug_port}/json/list");
    anyhow::ensure!(
        url.starts_with("http://127.0.0.1:"),
        "devtools helper restricted to loopback"
    );
    Ok(reqwest::get(url.as_str()).await?.json().await?)
}
