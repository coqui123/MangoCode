use once_cell::sync::Lazy;
use std::env;
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::warn;

/// Arguments passed to `ollama` when MangoCode autostarts a local server.
///
/// `ollama serve` reads its listen address from `OLLAMA_HOST`; passing
/// `--port` is rejected by every released `ollama` build, so the autostart
/// args must be exactly `["serve"]`.
pub const OLLAMA_AUTOSTART_ARGS: &[&str] = &["serve"];

struct OllamaProcess {
    child: Child,
}

impl Drop for OllamaProcess {
    fn drop(&mut self) {
        if let Err(err) = self.child.kill() {
            warn!(error = %err, "Failed to kill local Ollama process on shutdown");
        }
        let _ = self.child.wait();
    }
}

static LOCAL_OLLAMA_PROCESS: Lazy<Mutex<Option<Arc<OllamaProcess>>>> =
    Lazy::new(|| Mutex::new(None));

fn local_ollama_port_is_open() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], 11434));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

pub fn ensure_local_ollama_server() {
    if env::var_os("OLLAMA_HOST").is_some() {
        return;
    }
    if local_ollama_port_is_open() {
        return;
    }

    let mut guard = LOCAL_OLLAMA_PROCESS.lock().unwrap();
    if guard.is_some() {
        return;
    }

    // `ollama serve` does not accept a `--port` flag; the listen address is
    // controlled exclusively via the `OLLAMA_HOST` environment variable.
    // Earlier revisions passed `--port 11434`, which made the spawn fail
    // immediately on every host. We now invoke `ollama serve` cleanly and let
    // the user override the address via `OLLAMA_HOST` (we leave the default
    // `127.0.0.1:11434` alone here so the connection probe above matches).
    let child = match Command::new("ollama")
        .args(OLLAMA_AUTOSTART_ARGS)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            warn!(error = %err, "Failed to spawn local Ollama server");
            return;
        }
    };

    let process = Arc::new(OllamaProcess { child });
    *guard = Some(process.clone());

    for _ in 0..15 {
        if local_ollama_port_is_open() {
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    warn!("Local Ollama server did not become available after spawn");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autostart_args_do_not_include_invalid_port_flag() {
        // `ollama serve --port` is invalid — see the comment in
        // `ensure_local_ollama_server`. Guard against re-introducing it.
        assert!(
            !OLLAMA_AUTOSTART_ARGS.contains(&"--port"),
            "--port is not a valid `ollama serve` flag; configure via OLLAMA_HOST"
        );
        assert_eq!(OLLAMA_AUTOSTART_ARGS, &["serve"]);
    }
}
