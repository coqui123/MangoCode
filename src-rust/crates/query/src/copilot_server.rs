//! Auto-start the local Copilot proxy (`server.py` from the Copilot-pirate
//! checkout) when MangoCode's `lmstudio` slot is pointed at it but it isn't
//! running yet.
//!
//! The proxy location is resolved from (highest priority first):
//! - `COPILOT_SERVER_SCRIPT` — full path to `server.py`
//! - `COPILOT_SERVER_DIR` — directory containing `server.py`
//! - `config.env` entries for the keys above
//! - `provider_configs.lmstudio.options.copilot_server_script` /
//!   `copilot_server_dir`
//!
//! The server is spawned detached so it persists across MangoCode sessions — no
//! re-spawn on every launch. No-op when the port is already open, the path isn't
//! configured, or a spawn was already attempted this run.
//!
//! Also handles automatic token refresh on startup and periodic refresh every
//! 45 minutes via the proxy's `/v1/token/refresh` endpoint.

use std::env;
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use tracing::{debug, info, warn};

use mangocode_core::Config;

static SPAWN_ATTEMPTED: AtomicBool = AtomicBool::new(false);
static TOKEN_REFRESH_STARTED: AtomicBool = AtomicBool::new(false);

const TOKEN_REFRESH_INTERVAL_SECS: u64 = 45 * 60;

/// The Copilot proxy port (`COPILOT_API_PORT`, default 8765).
pub fn copilot_port() -> u16 {
    env::var("COPILOT_API_PORT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(8765)
}

fn port_is_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

struct ServerLocation {
    dir: PathBuf,
    script: PathBuf,
}

static RESOLVED_LOCATION: OnceLock<ServerLocation> = OnceLock::new();

fn non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn env_value(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .and_then(|v| non_empty(&v).map(str::to_string))
}

fn config_env_value(config: &Config, key: &str) -> Option<String> {
    config
        .env
        .get(key)
        .and_then(|v| non_empty(v).map(str::to_string))
}

fn provider_option_string(config: &Config, key: &str) -> Option<String> {
    for provider_id in ["lmstudio", "lm-studio"] {
        let Some(pc) = config.lookup_provider_config(provider_id) else {
            continue;
        };
        let Some(raw) = pc.options.get(key) else {
            continue;
        };
        if let Some(value) = raw.as_str().and_then(non_empty) {
            return Some(value.to_string());
        }
    }
    None
}

fn location_from_script_path(script: PathBuf) -> Option<ServerLocation> {
    if !script.is_file() {
        return None;
    }
    let dir = script.parent()?.to_path_buf();
    Some(ServerLocation { dir, script })
}

fn location_from_dir(dir: PathBuf) -> Option<ServerLocation> {
    let script = dir.join("server.py");
    script.is_file().then_some(ServerLocation { dir, script })
}

fn resolve_server_location_from_sources(
    script: Option<String>,
    dir: Option<String>,
) -> Option<ServerLocation> {
    if let Some(path) = script {
        if let Some(loc) = location_from_script_path(PathBuf::from(path)) {
            return Some(loc);
        }
    }
    dir.and_then(|path| location_from_dir(PathBuf::from(path)))
}

fn resolve_server_location(config: Option<&Config>) -> Option<ServerLocation> {
    if let Some(loc) = RESOLVED_LOCATION.get() {
        return Some(ServerLocation {
            dir: loc.dir.clone(),
            script: loc.script.clone(),
        });
    }

    let script = env_value("COPILOT_SERVER_SCRIPT").or_else(|| {
        config.and_then(|cfg| {
            config_env_value(cfg, "COPILOT_SERVER_SCRIPT")
                .or_else(|| provider_option_string(cfg, "copilot_server_script"))
        })
    });
    let dir = env_value("COPILOT_SERVER_DIR").or_else(|| {
        config.and_then(|cfg| {
            config_env_value(cfg, "COPILOT_SERVER_DIR")
                .or_else(|| provider_option_string(cfg, "copilot_server_dir"))
        })
    });

    let loc = resolve_server_location_from_sources(script, dir)?;

    // Persist for background refresh threads and later calls in this process.
    if env::var("COPILOT_SERVER_DIR").is_err() {
        env::set_var("COPILOT_SERVER_DIR", &loc.dir);
    }
    if env::var("COPILOT_SERVER_SCRIPT").is_err() {
        env::set_var("COPILOT_SERVER_SCRIPT", &loc.script);
    }
    let _ = RESOLVED_LOCATION.set(ServerLocation {
        dir: loc.dir.clone(),
        script: loc.script.clone(),
    });
    Some(loc)
}

/// Resolve the Copilot-pirate checkout dir. Returns `None` unless configured.
fn server_dir(config: Option<&Config>) -> Option<PathBuf> {
    resolve_server_location(config).map(|loc| loc.dir)
}

fn venv_python(dir: &Path) -> PathBuf {
    for rel in [
        dir.join(".venv").join("Scripts").join("python.exe"), // Windows
        dir.join(".venv").join("bin").join("python"),         // POSIX
    ] {
        if rel.exists() {
            return rel;
        }
    }
    PathBuf::from("python")
}

/// Hit the proxy's `/v1/token/refresh` endpoint to force a fresh substrate
/// token. Uses a raw HTTP POST over `TcpStream` to avoid pulling in reqwest
/// for this one blocking call. Returns `true` on success.
pub fn refresh_copilot_token() -> bool {
    let port = copilot_port();
    if !port_is_open(port) {
        debug!("Copilot proxy not listening on port {port}; skipping token refresh");
        return false;
    }

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(5)) else {
        warn!("Failed to connect to Copilot proxy for token refresh");
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let request = format!(
        "POST /v1/token/refresh HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Length: 0\r\n\
         Connection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        warn!("Failed to send token refresh request");
        return false;
    }
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    let response = String::from_utf8_lossy(&buf);
    if response.contains("200") {
        info!("Copilot token refreshed successfully");
        true
    } else {
        warn!(response = %response.chars().take(200).collect::<String>(), "Copilot token refresh failed");
        false
    }
}

/// The CDP debug port used by the browser token extractor.
const CDP_DEBUG_PORT: u16 = 9225;

/// Check whether an Edge CDP session is already listening on the debug port.
fn cdp_session_alive() -> bool {
    port_is_open(CDP_DEBUG_PORT)
}

/// Run the browser token extractor (`extract_sydney_browser.py`) to get a fresh
/// WebSocket-capable Sydney token. Reuses an existing Edge CDP session on port
/// 9225 if one is already running, avoiding redundant browser launches.
/// Returns `true` if the extraction succeeded.
pub fn refresh_browser_token() -> bool {
    let Some(dir) = server_dir(None) else {
        debug!("Copilot server path not configured; cannot run browser token refresh");
        return false;
    };
    let script = dir.join("extract_sydney_browser.py");
    if !script.exists() {
        debug!("extract_sydney_browser.py not found; skipping browser token refresh");
        return false;
    }
    let py = venv_python(&dir);
    let mut cmd = Command::new(&py);
    cmd.arg("extract_sydney_browser.py")
        .arg("--account-hint")
        .arg("nvlx")
        .arg("--debug-port")
        .arg(CDP_DEBUG_PORT.to_string())
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if cdp_session_alive() {
        debug!("Reusing existing Edge CDP session on port {CDP_DEBUG_PORT}");
    } else {
        debug!("No existing CDP session; extract_sydney_browser.py will launch Edge");
        cmd.arg("--pre-message-sleep").arg("15");
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(mut child) => {
            info!("Launched browser token extractor (extract_sydney_browser.py)");
            // Wait up to 120s for it to complete so the token is ready.
            match child.wait() {
                Ok(status) if status.success() => {
                    info!("Browser token extraction completed successfully");
                    true
                }
                Ok(status) => {
                    warn!(exit_code = ?status.code(), "Browser token extraction failed");
                    false
                }
                Err(e) => {
                    warn!(error = %e, "Failed to wait for browser token extraction");
                    false
                }
            }
        }
        Err(err) => {
            warn!(error = %err, "Failed to spawn extract_sydney_browser.py");
            false
        }
    }
}

/// Spawn a background thread that refreshes the Copilot token every 45 minutes.
/// Called once per process; subsequent calls are no-ops.
pub fn start_periodic_token_refresh() {
    if TOKEN_REFRESH_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::Builder::new()
        .name("copilot-token-refresh".into())
        .spawn(|| {
            loop {
                std::thread::sleep(Duration::from_secs(TOKEN_REFRESH_INTERVAL_SECS));
                debug!("Periodic Copilot token refresh tick");
                if !port_is_open(copilot_port()) {
                    debug!("Copilot proxy not running; skipping periodic refresh");
                    continue;
                }
                // Refresh substrate token first, then get a fresh browser token.
                refresh_copilot_token();
                refresh_browser_token();
            }
        })
        .ok();
}

/// Start `server.py` if the Copilot proxy port isn't already listening.
pub fn ensure_copilot_server_from_config(config: &Config) {
    let port = copilot_port();
    if port_is_open(port) {
        return;
    }
    // One spawn attempt per process — avoids storms from repeated turns.
    if SPAWN_ATTEMPTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(loc) = resolve_server_location(Some(config)) else {
        debug!(
            "Copilot proxy not running and no server path is configured \
             (COPILOT_SERVER_SCRIPT, COPILOT_SERVER_DIR, config.env, or \
             provider_configs.lmstudio.options); skipping autostart"
        );
        return;
    };
    let dir = &loc.dir;

    let py = venv_python(dir);
    let mut cmd = Command::new(&py);
    cmd.arg(
        loc.script
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "server.py".to_string()),
    )
    .current_dir(dir)
    .stdin(Stdio::null());
    // Route output to a log file so we don't inherit the TUI's stdio.
    match std::fs::File::create(dir.join("server.autostart.log")) {
        Ok(file) => {
            let err = file.try_clone().ok();
            cmd.stdout(Stdio::from(file));
            match err {
                Some(e) => {
                    cmd.stderr(Stdio::from(e));
                }
                None => {
                    cmd.stderr(Stdio::null());
                }
            }
        }
        Err(_) => {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }
    // Detach so the server survives MangoCode exiting (and so Ctrl-C in the TUI
    // doesn't tear it down).
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(_child) => {
            // Intentionally do not retain the Child: dropping it must NOT kill
            // the proxy, so it keeps running for the next session too.
            debug!(port, dir = %dir.display(), script = %loc.script.display(), "Spawning Copilot proxy");
            for _ in 0..40 {
                if port_is_open(port) {
                    debug!(port, "Copilot proxy is up");
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            warn!(
                port,
                "Copilot proxy did not come up within timeout after autostart"
            );
        }
        Err(err) => warn!(error = %err, "Failed to spawn Copilot proxy server.py"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copilot_port_honors_env_and_default() {
        std::env::remove_var("COPILOT_API_PORT");
        assert_eq!(copilot_port(), 8765);
        std::env::set_var("COPILOT_API_PORT", "9001");
        assert_eq!(copilot_port(), 9001);
        std::env::remove_var("COPILOT_API_PORT");
    }

    #[test]
    fn server_dir_requires_server_py() {
        // Unset / bogus dir resolves to None (no autostart).
        std::env::remove_var("COPILOT_SERVER_SCRIPT");
        std::env::set_var("COPILOT_SERVER_DIR", "/nonexistent/copilot/dir");
        assert!(server_dir(None).is_none());
        std::env::remove_var("COPILOT_SERVER_DIR");
        assert!(server_dir(None).is_none());
    }

    #[test]
    fn resolve_server_script_path() {
        std::env::remove_var("COPILOT_SERVER_DIR");
        std::env::remove_var("COPILOT_SERVER_SCRIPT");
        let dir = std::env::temp_dir().join(format!(
            "mangocode-copilot-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("server.py");
        std::fs::write(&script, "# stub\n").unwrap();

        assert!(resolve_server_location_from_sources(
            Some(script.display().to_string()),
            None
        )
        .is_some());
        assert!(resolve_server_location_from_sources(
            None,
            Some(dir.display().to_string())
        )
        .is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
