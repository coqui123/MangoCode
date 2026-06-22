//! Placeholder for nodriver-style authenticated proxy forwarding (`ProxyForwarder`).

#[allow(dead_code)]
pub fn auth_socks_upstream_not_implemented() -> Result<(), &'static str> {
    Err(
        "Authenticated HTTP/S SOCKS proxy bridging is not implemented in MangoCode. \
         Use Chromium --proxy-server without embedded credentials, or configure an external forwarder.",
    )
}
