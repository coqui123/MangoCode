use once_cell::sync::Lazy;
use std::env;
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::warn;

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

    let child = match Command::new("ollama")
        .args(["serve", "--port", "11434"])
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
