// cc-mcp: Model Context Protocol (MCP) client implementation.
//
// MCP is a JSON-RPC 2.0 based protocol for connecting Claude to external
// tool/resource servers. This crate implements:
//
// - JSON-RPC 2.0 client primitives
// - MCP protocol handshake (initialize, initialized)
// - Tool discovery (tools/list)
// - Tool execution (tools/call)
// - Resource management (resources/list, resources/read)
// - Prompt templates (prompts/list, prompts/get)
// - Transport: stdio (subprocess) and HTTP/SSE
// - Environment variable expansion in server configs
// - Connection manager with exponential-backoff reconnection

use async_trait::async_trait;
use dashmap::DashMap;
use futures::stream::{BoxStream, StreamExt};
use mangocode_core::config::{McpServerConfig, PipedreamMcpConfig};
use mangocode_core::mcp_templates::TemplateRenderer;
use mangocode_core::types::ToolDefinition;
use mangocode_core::vault::PipedreamConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

pub use client::McpClient;
pub use connection_manager::{McpConnectionManager, McpServerStatus};
pub use types::*;

pub mod connection_manager;
pub mod oauth;
pub mod registry;

// ---------------------------------------------------------------------------
// Environment variable expansion
// ---------------------------------------------------------------------------

/// Expand `${VAR_NAME}` and `${VAR_NAME:-default}` patterns in `input` using
/// the process environment.  Unknown variables without a default are left as-is
/// (matching the TS behaviour: report missing but don't crash).
pub fn expand_env_vars(input: &str) -> String {
    let mut result = input.to_string();
    // We iterate from left to right, always restarting the search after each
    // substitution so that replaced values are not re-scanned.
    let mut search_from = 0;
    loop {
        match result[search_from..].find("${") {
            None => break,
            Some(rel_start) => {
                let start = search_from + rel_start;
                match result[start..].find('}') {
                    None => break, // unclosed brace — stop
                    Some(rel_end) => {
                        let end = start + rel_end; // index of '}'
                        let inner = &result[start + 2..end]; // content between ${ and }

                        // Support ${VAR:-default} syntax
                        let (var_name, default_value) = if let Some(pos) = inner.find(":-") {
                            (&inner[..pos], Some(&inner[pos + 2..]))
                        } else {
                            (inner, None)
                        };

                        let replacement = match std::env::var(var_name) {
                            Ok(val) => val,
                            Err(_) => match default_value {
                                Some(def) => def.to_string(),
                                None => {
                                    // Leave the original text in place; advance past it
                                    search_from = end + 1;
                                    continue;
                                }
                            },
                        };

                        result =
                            format!("{}{}{}", &result[..start], replacement, &result[end + 1..]);
                        // Continue scanning from where the replacement ends
                        search_from = start + replacement.len();
                    }
                }
            }
        }
    }
    result
}

/// Expand env vars in every string field of a `McpServerConfig`.
/// Returns a new owned config; the original is not modified.
pub fn expand_server_config(config: &McpServerConfig) -> McpServerConfig {
    McpServerConfig {
        name: config.name.clone(),
        command: config.command.as_deref().map(expand_env_vars),
        args: config.args.iter().map(|a| expand_env_vars(a)).collect(),
        env: config
            .env
            .iter()
            .map(|(k, v)| (k.clone(), expand_env_vars(v)))
            .collect(),
        url: config.url.as_deref().map(expand_env_vars),
        headers: config
            .headers
            .iter()
            .map(|(k, v)| (k.clone(), expand_env_vars(v)))
            .collect(),
        pipedream: config.pipedream.as_ref().map(expand_pipedream_config),
        server_type: config.server_type.clone(),
    }
}

fn expand_optional_env(input: &Option<String>) -> Option<String> {
    input.as_deref().map(expand_env_vars)
}

fn expand_pipedream_config(config: &PipedreamMcpConfig) -> PipedreamMcpConfig {
    PipedreamMcpConfig {
        client_id: expand_optional_env(&config.client_id),
        client_secret: expand_optional_env(&config.client_secret),
        project_id: expand_optional_env(&config.project_id),
        environment: expand_optional_env(&config.environment),
        external_user_id: expand_optional_env(&config.external_user_id),
        app_slug: expand_optional_env(&config.app_slug),
        app_discovery: config.app_discovery,
        account_id: expand_optional_env(&config.account_id),
        tool_mode: expand_optional_env(&config.tool_mode),
        conversation_id: expand_optional_env(&config.conversation_id),
        scope: expand_optional_env(&config.scope),
        token_url: expand_optional_env(&config.token_url),
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 Types
// ---------------------------------------------------------------------------

pub mod types {
    use super::*;

    /// A JSON-RPC 2.0 request.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcRequest {
        pub jsonrpc: String,
        pub id: Value,
        pub method: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub params: Option<Value>,
    }

    impl JsonRpcRequest {
        pub fn new(id: impl Into<Value>, method: impl Into<String>, params: Option<Value>) -> Self {
            Self {
                jsonrpc: "2.0".to_string(),
                id: id.into(),
                method: method.into(),
                params,
            }
        }

        pub fn notification(method: impl Into<String>, params: Option<Value>) -> Self {
            Self {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                method: method.into(),
                params,
            }
        }
    }

    /// A JSON-RPC 2.0 response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcResponse {
        pub jsonrpc: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<JsonRpcError>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcError {
        pub code: i64,
        pub message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<Value>,
    }

    // ---- MCP protocol types ------------------------------------------------

    /// MCP initialize request params.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct InitializeParams {
        pub protocol_version: String,
        pub capabilities: ClientCapabilities,
        pub client_info: ClientInfo,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ClientCapabilities {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub roots: Option<RootsCapability>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sampling: Option<Value>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RootsCapability {
        #[serde(rename = "listChanged")]
        pub list_changed: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ClientInfo {
        pub name: String,
        pub version: String,
    }

    /// MCP initialize response result.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct InitializeResult {
        pub protocol_version: String,
        pub capabilities: ServerCapabilities,
        pub server_info: ServerInfo,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub instructions: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ServerCapabilities {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<ToolsCapability>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resources: Option<ResourcesCapability>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompts: Option<PromptsCapability>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logging: Option<Value>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolsCapability {
        #[serde(default)]
        pub list_changed: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ResourcesCapability {
        #[serde(default)]
        pub subscribe: bool,
        #[serde(default)]
        pub list_changed: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptsCapability {
        #[serde(default)]
        pub list_changed: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ServerInfo {
        pub name: String,
        pub version: String,
    }

    /// An MCP tool definition.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpTool {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        pub input_schema: Value,
    }

    impl From<&McpTool> for ToolDefinition {
        fn from(t: &McpTool) -> Self {
            ToolDefinition {
                name: t.name.clone(),
                description: t.description.clone().unwrap_or_default(),
                input_schema: t.input_schema.clone(),
            }
        }
    }

    /// tools/list response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ListToolsResult {
        pub tools: Vec<McpTool>,
        #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    /// tools/call params.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CallToolParams {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<Value>,
    }

    /// tools/call response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CallToolResult {
        pub content: Vec<McpContent>,
        #[serde(default)]
        pub is_error: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum McpContent {
        Text {
            text: String,
        },
        Image {
            data: String,
            #[serde(rename = "mimeType")]
            mime_type: String,
        },
        Resource {
            resource: ResourceContents,
        },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ResourceContents {
        pub uri: String,
        #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub blob: Option<String>,
    }

    /// An MCP resource.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpResource {
        pub uri: String,
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Value>,
    }

    /// resources/list response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ListResourcesResult {
        pub resources: Vec<McpResource>,
        #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
        pub next_cursor: Option<String>,
    }

    /// An MCP prompt template.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpPrompt {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(default)]
        pub arguments: Vec<McpPromptArgument>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpPromptArgument {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(default)]
        pub required: bool,
    }

    /// prompts/list response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ListPromptsResult {
        pub prompts: Vec<McpPrompt>,
    }

    /// A single message returned by prompts/get.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PromptMessage {
        /// "user" or "assistant"
        pub role: String,
        pub content: PromptMessageContent,
    }

    /// Content inside a PromptMessage.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum PromptMessageContent {
        Text { text: String },
        Image { data: String, mime_type: String },
        Resource { resource: serde_json::Value },
    }

    /// prompts/get response.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetPromptResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        pub messages: Vec<PromptMessage>,
    }
}

// ---------------------------------------------------------------------------
// Transport layer
// ---------------------------------------------------------------------------

pub mod transport {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
    use std::str::FromStr;

    /// A transport can send requests and receive responses.
    #[async_trait]
    pub trait McpTransport: Send + Sync {
        async fn send(&self, message: &JsonRpcRequest) -> anyhow::Result<()>;
        async fn recv(&self) -> anyhow::Result<Option<JsonRpcResponse>>;
        async fn close(&self) -> anyhow::Result<()>;
        async fn set_protocol_version(&self, _version: Option<String>) -> anyhow::Result<()> {
            Ok(())
        }
        /// Non-blocking poll: return the next raw JSON message if one is
        /// immediately available, or `Ok(None)` if the queue is empty.
        /// Used by the notification dispatch loop to drain server-initiated
        /// notifications without blocking an async task.
        async fn try_receive_raw(&self) -> anyhow::Result<Option<serde_json::Value>>;
        /// Subscribe to raw JSON notifications from the transport.
        /// Returns an async stream of notification messages.
        ///
        /// For transports that natively support push notifications (e.g., WebSocket),
        /// this returns a stream that yields messages directly from the transport.
        /// For transports without native push support (e.g., stdio), this returns
        /// a stream that polls periodically.
        fn subscribe_to_notifications(
            &self,
        ) -> BoxStream<'static, anyhow::Result<serde_json::Value>>;
    }

    fn parse_raw_message(line: &str) -> anyhow::Result<serde_json::Value> {
        serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("MCP raw parse error: {} (raw: {})", e, line))
    }

    fn parse_sse_or_json_messages(body: &str) -> anyhow::Result<Vec<serde_json::Value>> {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Ok(vec![parse_raw_message(trimmed)?]);
        }

        let mut messages = Vec::new();
        let mut event_lines = Vec::new();

        for line in body.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                if !event_lines.is_empty() {
                    let joined = event_lines.join("\n");
                    messages.push(parse_raw_message(joined.trim())?);
                    event_lines.clear();
                }
                continue;
            }

            if let Some(data) = line.strip_prefix("data:") {
                event_lines.push(data.trim_start().to_string());
            }
        }

        if !event_lines.is_empty() {
            let joined = event_lines.join("\n");
            messages.push(parse_raw_message(joined.trim())?);
        }

        if messages.is_empty() {
            anyhow::bail!("MCP remote transport returned an unsupported response body");
        }

        Ok(messages)
    }

    /// Stdio transport: spawns a subprocess and communicates via stdin/stdout.
    pub struct StdioTransport {
        child: Arc<Mutex<Child>>,
        stdin: Arc<Mutex<ChildStdin>>,
        stdout_rx: Arc<Mutex<mpsc::UnboundedReceiver<serde_json::Value>>>,
    }

    impl StdioTransport {
        pub async fn spawn(config: &McpServerConfig) -> anyhow::Result<Self> {
            let command = config.command.as_deref().ok_or_else(|| {
                anyhow::anyhow!("MCP server '{}' has no command configured", config.name)
            })?;

            let mut cmd = Command::new(command);
            cmd.args(&config.args)
                .envs(&config.env)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                anyhow::anyhow!(
                    "MCP server '{}': failed to spawn '{}': {}",
                    config.name,
                    command,
                    e
                )
            })?;

            let stdin = child.stdin.take().ok_or_else(|| {
                anyhow::anyhow!("MCP server '{}': could not capture stdin", config.name)
            })?;
            let stdout = child.stdout.take().ok_or_else(|| {
                anyhow::anyhow!("MCP server '{}': could not capture stdout", config.name)
            })?;

            let (tx, rx) = mpsc::unbounded_channel::<serde_json::Value>();

            // Background reader task — forwards stdout lines to the channel.
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    match parse_raw_message(&line) {
                        Ok(value) => {
                            if tx.send(value).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to parse MCP stdio message");
                        }
                    }
                }
            });

            Ok(Self {
                child: Arc::new(Mutex::new(child)),
                stdin: Arc::new(Mutex::new(stdin)),
                stdout_rx: Arc::new(Mutex::new(rx)),
            })
        }
    }

    #[async_trait]
    impl McpTransport for StdioTransport {
        async fn send(&self, message: &JsonRpcRequest) -> anyhow::Result<()> {
            let json = serde_json::to_string(message)? + "\n";
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(json.as_bytes()).await?;
            stdin.flush().await?;
            Ok(())
        }

        async fn recv(&self) -> anyhow::Result<Option<JsonRpcResponse>> {
            let mut rx = self.stdout_rx.lock().await;
            match rx.recv().await {
                Some(value) => {
                    let resp: JsonRpcResponse =
                        serde_json::from_value(value.clone()).map_err(|e| {
                            anyhow::anyhow!("MCP response parse error: {} (raw: {})", e, value)
                        })?;
                    Ok(Some(resp))
                }
                None => Ok(None),
            }
        }

        async fn close(&self) -> anyhow::Result<()> {
            let mut child = self.child.lock().await;
            let _ = child.kill().await;
            Ok(())
        }

        async fn try_receive_raw(&self) -> anyhow::Result<Option<serde_json::Value>> {
            let mut rx = self.stdout_rx.lock().await;
            match rx.try_recv() {
                Ok(value) => Ok(Some(value)),
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => Ok(None),
            }
        }

        fn subscribe_to_notifications(
            &self,
        ) -> BoxStream<'static, anyhow::Result<serde_json::Value>> {
            let stdout_rx = Arc::clone(&self.stdout_rx);

            // Create a channel to bridge from the exclusive receiver to the stream
            let (tx, rx) = mpsc::channel::<anyhow::Result<serde_json::Value>>(100);

            // Spawn a background task that polls the stdout_rx and forwards to tx
            tokio::spawn(async move {
                loop {
                    let line = {
                        let mut out_rx = stdout_rx.lock().await;
                        out_rx.recv().await
                    };

                    match line {
                        Some(value) => {
                            if tx.send(Ok(value)).await.is_err() {
                                // Receiver dropped; exit the polling task
                                break;
                            }
                        }
                        None => {
                            // stdout_rx closed
                            break;
                        }
                    }
                }
            });

            Box::pin(ReceiverStream::new(rx))
        }
    }

    /// Remote HTTP/SSE transport for hosted MCP servers.
    pub struct HttpTransport {
        client: reqwest::Client,
        url: String,
        headers: HeaderMap,
        pipedream: Option<PipedreamMcpConfig>,
        pipedream_token: Arc<Mutex<Option<PipedreamAccessToken>>>,
        session_id: Arc<Mutex<Option<String>>>,
        protocol_version: Arc<Mutex<Option<String>>>,
        raw_rx: Arc<Mutex<mpsc::UnboundedReceiver<serde_json::Value>>>,
        raw_tx: mpsc::UnboundedSender<serde_json::Value>,
    }

    #[derive(Debug, Clone)]
    struct PipedreamAccessToken {
        access_token: String,
        expires_at: u64,
    }

    impl HttpTransport {
        fn normalize_optional_string(value: Option<String>) -> Option<String> {
            value.and_then(|value| {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
        }

        fn read_env(key: &str) -> Option<String> {
            Self::normalize_optional_string(std::env::var(key).ok())
        }

        fn read_vault_secret(keys: &[&str]) -> Option<String> {
            let vault = mangocode_core::Vault::new();
            let passphrase = mangocode_core::get_vault_passphrase()?;
            for key in keys {
                if let Ok(Some(value)) = vault.get_secret(key, &passphrase) {
                    if let Some(value) = Self::normalize_optional_string(Some(value)) {
                        return Some(value);
                    }
                }
            }
            None
        }

        fn parse_bool_value(value: &str) -> Option<bool> {
            match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        }

        fn prefer_pipedream_string(
            config_value: Option<String>,
            vault_value: Option<String>,
            env_value: Option<String>,
        ) -> Option<String> {
            Self::normalize_optional_string(config_value)
                .or(vault_value)
                .or(env_value)
        }

        fn prefer_pipedream_bool(
            config_value: Option<bool>,
            vault_value: Option<bool>,
            env_value: Option<bool>,
        ) -> Option<bool> {
            config_value.or(vault_value).or(env_value)
        }

        fn resolve_pipedream_string(
            config_value: Option<String>,
            env_var: &str,
            vault_keys: &[&str],
        ) -> Option<String> {
            Self::prefer_pipedream_string(
                config_value,
                Self::read_vault_secret(vault_keys),
                Self::read_env(env_var),
            )
        }

        fn resolve_pipedream_string_with_file(
            config_value: Option<String>,
            env_var: &str,
            vault_keys: &[&str],
            file_value: Option<String>,
        ) -> Option<String> {
            // Priority: explicit config > vault > env > pipedream.json fallback
            Self::prefer_pipedream_string(
                config_value,
                Self::read_vault_secret(vault_keys),
                Self::read_env(env_var),
            )
            .or_else(|| Self::normalize_optional_string(file_value))
        }

        fn resolve_pipedream_bool(
            config_value: Option<bool>,
            env_var: &str,
            vault_keys: &[&str],
        ) -> Option<bool> {
            Self::prefer_pipedream_bool(
                config_value,
                Self::read_vault_secret(vault_keys)
                    .and_then(|value| Self::parse_bool_value(&value)),
                Self::read_env(env_var).and_then(|value| Self::parse_bool_value(&value)),
            )
        }

        fn pipedream_vault_keys(field: &str) -> &'static [&'static str] {
            match field {
                "client_id" => &[
                    "pipedream-client-id",
                    "pipedream_client_id",
                    "PIPEDREAM_CLIENT_ID",
                ],
                "client_secret" => &[
                    "pipedream-client-secret",
                    "pipedream_client_secret",
                    "PIPEDREAM_CLIENT_SECRET",
                ],
                "project_id" => &[
                    "pipedream-project-id",
                    "pipedream_project_id",
                    "PIPEDREAM_PROJECT_ID",
                ],
                "environment" => &[
                    "pipedream-environment",
                    "pipedream_environment",
                    "PIPEDREAM_ENVIRONMENT",
                ],
                "external_user_id" => &[
                    "pipedream-external-user-id",
                    "pipedream_external_user_id",
                    "PIPEDREAM_EXTERNAL_USER_ID",
                ],
                "app_slug" => &[
                    "pipedream-app-slug",
                    "pipedream_app_slug",
                    "PIPEDREAM_APP_SLUG",
                ],
                "app_discovery" => &[
                    "pipedream-app-discovery",
                    "pipedream_app_discovery",
                    "PIPEDREAM_APP_DISCOVERY",
                ],
                "account_id" => &[
                    "pipedream-account-id",
                    "pipedream_account_id",
                    "PIPEDREAM_ACCOUNT_ID",
                ],
                "tool_mode" => &[
                    "pipedream-tool-mode",
                    "pipedream_tool_mode",
                    "PIPEDREAM_TOOL_MODE",
                ],
                "conversation_id" => &[
                    "pipedream-conversation-id",
                    "pipedream_conversation_id",
                    "PIPEDREAM_CONVERSATION_ID",
                ],
                "scope" => &["pipedream-scope", "pipedream_scope", "PIPEDREAM_SCOPE"],
                "token_url" => &[
                    "pipedream-token-url",
                    "pipedream_token_url",
                    "PIPEDREAM_TOKEN_URL",
                ],
                "mcp_url" => &[
                    "pipedream-mcp-url",
                    "pipedream_mcp_url",
                    "PIPEDREAM_MCP_URL",
                ],
                _ => &[],
            }
        }

        pub async fn connect(config: &McpServerConfig) -> anyhow::Result<Self> {
            let url = if config.server_type == "pipedream" {
                // Priority: explicit config url > vault > env > pipedream.json fallback > default
                let explicit = config.url.as_deref().and_then(|u| {
                    let trimmed = u.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed)
                    }
                });
                if let Some(u) = explicit {
                    u.to_string()
                } else if let Some(vault_url) =
                    Self::read_vault_secret(Self::pipedream_vault_keys("mcp_url"))
                {
                    vault_url
                } else if let Some(env_url) = Self::read_env("PIPEDREAM_MCP_URL") {
                    env_url
                } else if let Some(file_url) = PipedreamConfig::load()
                    .map(|c| c.mcp_url())
                    .filter(|s| !s.is_empty())
                {
                    file_url
                } else {
                    "https://remote.mcp.pipedream.net/v3".to_string()
                }
            } else {
                config.url.as_deref().unwrap_or("").trim().to_string()
            };

            if url.is_empty() {
                anyhow::bail!("MCP server '{}' has no URL configured", config.name);
            }

            let mut headers = HeaderMap::new();
            headers.insert(
                ACCEPT,
                HeaderValue::from_static("application/json, text/event-stream"),
            );
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            for (name, value) in &config.headers {
                let header_name = HeaderName::from_str(name).map_err(|e| {
                    anyhow::anyhow!(
                        "MCP server '{}': invalid HTTP header '{}': {}",
                        config.name,
                        name,
                        e
                    )
                })?;
                let header_value = HeaderValue::from_str(value).map_err(|e| {
                    anyhow::anyhow!(
                        "MCP server '{}': invalid value for header '{}': {}",
                        config.name,
                        name,
                        e
                    )
                })?;
                headers.insert(header_name, header_value);
            }

            let client = mangocode_core::reqwest_client_builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build MCP HTTP client: {}", e))?;
            let (raw_tx, raw_rx) = mpsc::unbounded_channel();

            Ok(Self {
                client,
                url: url.to_string(),
                headers,
                pipedream: Self::pipedream_config(config)?,
                pipedream_token: Arc::new(Mutex::new(None)),
                session_id: Arc::new(Mutex::new(None)),
                protocol_version: Arc::new(Mutex::new(None)),
                raw_rx: Arc::new(Mutex::new(raw_rx)),
                raw_tx,
            })
        }

        fn pipedream_config(
            config: &McpServerConfig,
        ) -> anyhow::Result<Option<PipedreamMcpConfig>> {
            if config.server_type != "pipedream" {
                return Ok(config.pipedream.clone());
            }

            let mut pipedream = config.pipedream.clone().unwrap_or_default();

            // Load optional pipedream.json fallback values.
            let file_config = PipedreamConfig::load();

            pipedream.client_id = Self::resolve_pipedream_string_with_file(
                pipedream.client_id.clone(),
                "PIPEDREAM_CLIENT_ID",
                Self::pipedream_vault_keys("client_id"),
                file_config
                    .as_ref()
                    .and_then(|c| c.client_id.clone())
                    .filter(|s| !s.is_empty()),
            );
            pipedream.client_secret = Self::resolve_pipedream_string_with_file(
                pipedream.client_secret.clone(),
                "PIPEDREAM_CLIENT_SECRET",
                Self::pipedream_vault_keys("client_secret"),
                file_config
                    .as_ref()
                    .and_then(|c| c.client_secret.clone())
                    .filter(|s| !s.is_empty()),
            );
            pipedream.project_id = Self::resolve_pipedream_string_with_file(
                pipedream.project_id.clone(),
                "PIPEDREAM_PROJECT_ID",
                Self::pipedream_vault_keys("project_id"),
                file_config
                    .as_ref()
                    .and_then(|c| c.project_id.clone())
                    .filter(|s| !s.is_empty()),
            );
            pipedream.environment = Self::resolve_pipedream_string_with_file(
                pipedream.environment.clone(),
                "PIPEDREAM_ENVIRONMENT",
                Self::pipedream_vault_keys("environment"),
                file_config
                    .as_ref()
                    .map(|c| c.environment.clone())
                    .filter(|s| !s.is_empty()),
            )
            .or_else(|| Some("development".to_string()));
            pipedream.external_user_id = Self::resolve_pipedream_string(
                pipedream.external_user_id.clone(),
                "PIPEDREAM_EXTERNAL_USER_ID",
                Self::pipedream_vault_keys("external_user_id"),
            )
            .or_else(|| Some("local-dev".to_string()));
            pipedream.app_slug = Self::resolve_pipedream_string(
                pipedream.app_slug.clone(),
                "PIPEDREAM_APP_SLUG",
                Self::pipedream_vault_keys("app_slug"),
            );
            pipedream.app_discovery = Self::resolve_pipedream_bool(
                pipedream.app_discovery,
                "PIPEDREAM_APP_DISCOVERY",
                Self::pipedream_vault_keys("app_discovery"),
            );
            pipedream.account_id = Self::resolve_pipedream_string_with_file(
                pipedream.account_id.clone(),
                "PIPEDREAM_ACCOUNT_ID",
                Self::pipedream_vault_keys("account_id"),
                file_config
                    .as_ref()
                    .and_then(|c| c.account_id.clone())
                    .filter(|s| !s.is_empty()),
            );
            pipedream.tool_mode = Self::resolve_pipedream_string(
                pipedream.tool_mode.clone(),
                "PIPEDREAM_TOOL_MODE",
                Self::pipedream_vault_keys("tool_mode"),
            );
            pipedream.conversation_id = Self::resolve_pipedream_string(
                pipedream.conversation_id.clone(),
                "PIPEDREAM_CONVERSATION_ID",
                Self::pipedream_vault_keys("conversation_id"),
            );
            pipedream.scope = Self::resolve_pipedream_string(
                pipedream.scope.clone(),
                "PIPEDREAM_SCOPE",
                Self::pipedream_vault_keys("scope"),
            );
            pipedream.token_url = Self::resolve_pipedream_string_with_file(
                pipedream.token_url.clone(),
                "PIPEDREAM_TOKEN_URL",
                Self::pipedream_vault_keys("token_url"),
                file_config
                    .as_ref()
                    .and_then(|c| c.token_url.clone())
                    .filter(|s| !s.is_empty()),
            );

            let has_app = pipedream
                .app_slug
                .as_ref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            let uses_app_discovery = pipedream.app_discovery.unwrap_or(false);

            if !has_app && !uses_app_discovery {
                pipedream.app_discovery = Some(true);
            }

            Ok(Some(pipedream))
        }

        fn required_pipedream_value<'a>(
            config: &'a PipedreamMcpConfig,
            value: &'a Option<String>,
            name: &str,
        ) -> anyhow::Result<&'a str> {
            value
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    let _ = config;
                    anyhow::anyhow!("Pipedream MCP is missing required setting '{}'", name)
                })
        }

        fn now_secs() -> u64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        }

        async fn pipedream_access_token(
            &self,
            config: &PipedreamMcpConfig,
        ) -> anyhow::Result<String> {
            {
                let cached = self.pipedream_token.lock().await;
                if let Some(token) = cached.as_ref() {
                    if token.expires_at > Self::now_secs() + 60 {
                        return Ok(token.access_token.clone());
                    }
                }
            }

            let client_id = Self::required_pipedream_value(config, &config.client_id, "client_id")?;
            let client_secret =
                Self::required_pipedream_value(config, &config.client_secret, "client_secret")?;
            let token_url = config
                .token_url
                .as_deref()
                .unwrap_or("https://api.pipedream.com/v1/oauth/token");

            let mut payload = serde_json::json!({
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            });
            if let Some(scope) = config.scope.as_deref().filter(|s| !s.trim().is_empty()) {
                payload["scope"] = serde_json::Value::String(scope.to_string());
            }

            let response = self
                .client
                .post(token_url)
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .json(&payload)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Pipedream token request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Pipedream token request failed with {}: {}", status, body);
            }

            #[derive(Deserialize)]
            struct TokenResponse {
                access_token: String,
                expires_in: Option<u64>,
            }

            let token: TokenResponse = response
                .json()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to parse Pipedream token response: {}", e))?;
            let expires_at = Self::now_secs() + token.expires_in.unwrap_or(3600);

            let mut cached = self.pipedream_token.lock().await;
            *cached = Some(PipedreamAccessToken {
                access_token: token.access_token.clone(),
                expires_at,
            });

            Ok(token.access_token)
        }

        async fn request_headers(&self) -> anyhow::Result<HeaderMap> {
            let mut headers = self.headers.clone();

            if let Some(session_id) = self.session_id.lock().await.clone() {
                headers.insert(
                    HeaderName::from_static("mcp-session-id"),
                    HeaderValue::from_str(&session_id)?,
                );
            }

            if let Some(protocol_version) = self.protocol_version.lock().await.clone() {
                headers.insert(
                    HeaderName::from_static("mcp-protocol-version"),
                    HeaderValue::from_str(&protocol_version)?,
                );
            }

            if let Some(pipedream) = &self.pipedream {
                let token = self.pipedream_access_token(pipedream).await?;
                let authorization = format!("Bearer {}", token);
                headers.insert(
                    HeaderName::from_static("authorization"),
                    HeaderValue::from_str(&authorization)
                        .map_err(|e| anyhow::anyhow!("Invalid Pipedream access token: {}", e))?,
                );

                let project_id =
                    Self::required_pipedream_value(pipedream, &pipedream.project_id, "project_id")?;
                let environment = Self::required_pipedream_value(
                    pipedream,
                    &pipedream.environment,
                    "environment",
                )?;
                let external_user_id = Self::required_pipedream_value(
                    pipedream,
                    &pipedream.external_user_id,
                    "external_user_id",
                )?;

                headers.insert(
                    HeaderName::from_static("x-pd-project-id"),
                    HeaderValue::from_str(project_id)?,
                );
                headers.insert(
                    HeaderName::from_static("x-pd-environment"),
                    HeaderValue::from_str(environment)?,
                );
                headers.insert(
                    HeaderName::from_static("x-pd-external-user-id"),
                    HeaderValue::from_str(external_user_id)?,
                );

                if let Some(account_id) = pipedream.account_id.as_deref().filter(|s| !s.is_empty())
                {
                    headers.insert(
                        HeaderName::from_static("x-pd-account-id"),
                        HeaderValue::from_str(account_id)?,
                    );
                }

                if let Some(tool_mode) = pipedream.tool_mode.as_deref().filter(|s| !s.is_empty()) {
                    headers.insert(
                        HeaderName::from_static("x-pd-tool-mode"),
                        HeaderValue::from_str(tool_mode)?,
                    );
                }

                if let Some(conversation_id) = pipedream
                    .conversation_id
                    .as_deref()
                    .filter(|s| !s.is_empty())
                {
                    headers.insert(
                        HeaderName::from_static("x-pd-conversation-id"),
                        HeaderValue::from_str(conversation_id)?,
                    );
                }

                if pipedream.app_discovery.unwrap_or(false) {
                    headers.insert(
                        HeaderName::from_static("x-pd-app-discovery"),
                        HeaderValue::from_static("true"),
                    );
                } else {
                    let app_slug =
                        Self::required_pipedream_value(pipedream, &pipedream.app_slug, "app_slug")?;
                    headers.insert(
                        HeaderName::from_static("x-pd-app-slug"),
                        HeaderValue::from_str(app_slug)?,
                    );
                }
            }

            Ok(headers)
        }
    }

    #[async_trait]
    impl McpTransport for HttpTransport {
        async fn send(&self, message: &JsonRpcRequest) -> anyhow::Result<()> {
            let response = self
                .client
                .post(&self.url)
                .headers(self.request_headers().await?)
                .json(message)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("MCP HTTP request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("MCP HTTP request failed with {}: {}", status, body);
            }

            if let Some(session_id) = response.headers().get("mcp-session-id") {
                if let Ok(session_id) = session_id.to_str() {
                    let mut stored = self.session_id.lock().await;
                    *stored = Some(session_id.to_string());
                }
            }

            let body = response
                .text()
                .await
                .map_err(|e| anyhow::anyhow!("Failed reading MCP HTTP response body: {}", e))?;

            for message in parse_sse_or_json_messages(&body)? {
                if self.raw_tx.send(message).is_err() {
                    anyhow::bail!("MCP HTTP response channel closed");
                }
            }

            Ok(())
        }

        async fn recv(&self) -> anyhow::Result<Option<JsonRpcResponse>> {
            let mut rx = self.raw_rx.lock().await;
            loop {
                match rx.recv().await {
                    Some(value) => {
                        if value.get("id").map(|v| !v.is_null()).unwrap_or(false) {
                            let resp: JsonRpcResponse = serde_json::from_value(value.clone())
                                .map_err(|e| {
                                    anyhow::anyhow!(
                                        "MCP HTTP response parse error: {} (raw: {})",
                                        e,
                                        value
                                    )
                                })?;
                            return Ok(Some(resp));
                        }
                    }
                    None => return Ok(None),
                }
            }
        }

        async fn close(&self) -> anyhow::Result<()> {
            Ok(())
        }

        async fn set_protocol_version(&self, version: Option<String>) -> anyhow::Result<()> {
            let mut stored = self.protocol_version.lock().await;
            *stored = version;
            Ok(())
        }

        async fn try_receive_raw(&self) -> anyhow::Result<Option<serde_json::Value>> {
            let mut rx = self.raw_rx.lock().await;
            match rx.try_recv() {
                Ok(value) => Ok(Some(value)),
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => Ok(None),
            }
        }

        fn subscribe_to_notifications(
            &self,
        ) -> BoxStream<'static, anyhow::Result<serde_json::Value>> {
            let raw_rx = Arc::clone(&self.raw_rx);
            let (tx, rx) = mpsc::channel::<anyhow::Result<serde_json::Value>>(100);

            tokio::spawn(async move {
                loop {
                    let message = {
                        let mut receiver = raw_rx.lock().await;
                        receiver.recv().await
                    };

                    match message {
                        Some(value) => {
                            if tx.send(Ok(value)).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            });

            Box::pin(ReceiverStream::new(rx))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::HttpTransport;

        #[test]
        fn pipedream_string_precedence_prefers_explicit_then_vault_then_env() {
            assert_eq!(
                HttpTransport::prefer_pipedream_string(
                    Some("config".to_string()),
                    Some("vault".to_string()),
                    Some("env".to_string()),
                )
                .as_deref(),
                Some("config")
            );
            assert_eq!(
                HttpTransport::prefer_pipedream_string(
                    None,
                    Some("vault".to_string()),
                    Some("env".to_string()),
                )
                .as_deref(),
                Some("vault")
            );
            assert_eq!(
                HttpTransport::prefer_pipedream_string(None, None, Some("env".to_string()))
                    .as_deref(),
                Some("env")
            );
        }

        #[test]
        fn pipedream_bool_precedence_prefers_explicit_then_vault_then_env() {
            assert_eq!(
                HttpTransport::prefer_pipedream_bool(Some(true), Some(false), Some(false)),
                Some(true)
            );
            assert_eq!(
                HttpTransport::prefer_pipedream_bool(None, Some(false), Some(true)),
                Some(false)
            );
            assert_eq!(
                HttpTransport::prefer_pipedream_bool(None, None, Some(true)),
                Some(true)
            );
        }

        #[test]
        fn pipedream_vault_keys_include_hyphenated_and_env_aliases() {
            let keys = HttpTransport::pipedream_vault_keys("client_id");
            assert!(keys.contains(&"pipedream-client-id"));
            assert!(keys.contains(&"PIPEDREAM_CLIENT_ID"));
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Client
// ---------------------------------------------------------------------------

pub mod client {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A fully initialized MCP client connected to a single server.
    pub struct McpClient {
        pub server_name: String,
        pub server_info: Option<ServerInfo>,
        pub capabilities: ServerCapabilities,
        pub tools: Vec<McpTool>,
        pub resources: Vec<McpResource>,
        pub prompts: Vec<McpPrompt>,
        pub instructions: Option<String>,
        transport: Arc<dyn transport::McpTransport>,
        next_id: AtomicU64,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
        notification_rx: Arc<Mutex<mpsc::UnboundedReceiver<serde_json::Value>>>,
    }

    impl McpClient {
        fn start_message_router(
            server_name: String,
            transport: Arc<dyn transport::McpTransport>,
            pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
            notification_tx: mpsc::UnboundedSender<serde_json::Value>,
        ) {
            tokio::spawn(async move {
                let mut stream = transport.subscribe_to_notifications();

                while let Some(result) = stream.next().await {
                    let raw = match result {
                        Ok(raw) => raw,
                        Err(e) => {
                            debug!(server = %server_name, error = %e, "MCP message router stopped");
                            break;
                        }
                    };

                    let has_id = raw.get("id").map(|v| !v.is_null()).unwrap_or(false);
                    if has_id {
                        if let Some(id) = raw.get("id").and_then(|v| v.as_u64()) {
                            if let Some(sender) = pending.lock().await.remove(&id) {
                                match serde_json::from_value::<JsonRpcResponse>(raw.clone()) {
                                    Ok(response) => {
                                        let _ = sender.send(response);
                                    }
                                    Err(e) => {
                                        debug!(
                                            server = %server_name,
                                            error = %e,
                                            response_id = id,
                                            "Failed to parse MCP response in router"
                                        );
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    if raw.get("method").is_some() && notification_tx.send(raw).is_err() {
                        break;
                    }
                }
            });
        }

        fn new_with_transport(
            config: &McpServerConfig,
            transport: Arc<dyn transport::McpTransport>,
        ) -> Self {
            let pending = Arc::new(Mutex::new(HashMap::new()));
            let (notification_tx, notification_rx) = mpsc::unbounded_channel();

            Self::start_message_router(
                config.name.clone(),
                Arc::clone(&transport),
                Arc::clone(&pending),
                notification_tx,
            );

            Self {
                server_name: config.name.clone(),
                server_info: None,
                capabilities: ServerCapabilities::default(),
                tools: vec![],
                resources: vec![],
                prompts: vec![],
                instructions: None,
                transport,
                next_id: AtomicU64::new(1),
                pending,
                notification_rx: Arc::new(Mutex::new(notification_rx)),
            }
        }

        /// Connect to an MCP server using stdio transport and complete the
        /// initialize handshake.  The `config` should already have env vars
        /// expanded via `expand_server_config`.
        pub async fn connect_stdio(config: &McpServerConfig) -> anyhow::Result<Self> {
            let transport = transport::StdioTransport::spawn(config).await?;
            let client = Self::new_with_transport(config, Arc::new(transport));
            client.initialize().await
        }

        /// Connect to an MCP server over remote HTTP/SSE and complete the
        /// initialize handshake.
        pub async fn connect_remote(config: &McpServerConfig) -> anyhow::Result<Self> {
            let transport = transport::HttpTransport::connect(config).await?;
            let client = Self::new_with_transport(config, Arc::new(transport));
            client.initialize().await
        }

        /// Send the MCP initialize handshake and discover capabilities.
        async fn initialize(mut self) -> anyhow::Result<Self> {
            let params = InitializeParams {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ClientCapabilities {
                    roots: Some(RootsCapability {
                        list_changed: false,
                    }),
                    sampling: None,
                },
                client_info: ClientInfo {
                    name: mangocode_core::constants::APP_NAME.to_string(),
                    version: mangocode_core::constants::APP_VERSION.to_string(),
                },
            };

            let result: InitializeResult = self
                .call("initialize", Some(serde_json::to_value(&params)?))
                .await
                .map_err(|e| {
                    anyhow::anyhow!("MCP server '{}' initialize failed: {}", self.server_name, e)
                })?;

            self.server_info = Some(result.server_info);
            self.instructions = result.instructions;
            self.capabilities = result.capabilities.clone();
            self.transport
                .set_protocol_version(Some(result.protocol_version.clone()))
                .await?;

            // Send initialized notification
            let notif = JsonRpcRequest::notification("notifications/initialized", None);
            self.transport.send(&notif).await?;

            // Discover tools if supported
            if result.capabilities.tools.is_some() {
                match self.list_tools().await {
                    Ok(tools) => self.tools = tools,
                    Err(e) => warn!(server = %self.server_name, error = %e, "Failed to list tools"),
                }
            }

            // Discover resources if supported
            if result.capabilities.resources.is_some() {
                match self.list_resources().await {
                    Ok(resources) => self.resources = resources,
                    Err(e) => {
                        warn!(server = %self.server_name, error = %e, "Failed to list resources")
                    }
                }
            }

            // Discover prompts if supported
            if result.capabilities.prompts.is_some() {
                match self.list_prompts().await {
                    Ok(prompts) => self.prompts = prompts,
                    Err(e) => {
                        warn!(server = %self.server_name, error = %e, "Failed to list prompts")
                    }
                }
            }

            Ok(self)
        }

        // ---- High-level API -----------------------------------------------

        pub async fn list_tools(&self) -> anyhow::Result<Vec<McpTool>> {
            let result: ListToolsResult = self.call("tools/list", None).await?;
            Ok(result.tools)
        }

        pub async fn call_tool(
            &self,
            name: &str,
            arguments: Option<Value>,
        ) -> anyhow::Result<CallToolResult> {
            let params = CallToolParams {
                name: name.to_string(),
                arguments,
            };
            self.call("tools/call", Some(serde_json::to_value(&params)?))
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "MCP server '{}': tool '{}' call failed: {}",
                        self.server_name,
                        name,
                        e
                    )
                })
        }

        pub async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>> {
            let result: ListResourcesResult = self.call("resources/list", None).await?;
            let mut resources = result.resources;

            // Apply template rendering for resources with prompt annotations
            for resource in &mut resources {
                if let Some(annotations) = &resource.annotations {
                    if let Some(prompt_template) = annotations.get("prompt") {
                        if let Some(template_str) = prompt_template.as_str() {
                            // Build context from resource fields
                            let context = serde_json::json!({
                                "uri": resource.uri,
                                "name": resource.name,
                                "description": resource.description,
                                "mimeType": resource.mime_type,
                            });

                            // Render the template and replace description
                            let rendered = TemplateRenderer::render(template_str, &context);
                            resource.description = Some(rendered);
                        }
                    }
                }
            }

            Ok(resources)
        }

        pub async fn read_resource(&self, uri: &str) -> anyhow::Result<ResourceContents> {
            let params = serde_json::json!({ "uri": uri });
            let result: Value = self.call("resources/read", Some(params)).await?;
            let contents = result
                .get("contents")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "MCP server '{}': no contents in resources/read response for '{}'",
                        self.server_name,
                        uri
                    )
                })?;
            Ok(serde_json::from_value(contents.clone())?)
        }

        pub async fn list_prompts(&self) -> anyhow::Result<Vec<McpPrompt>> {
            let result: ListPromptsResult = self.call("prompts/list", None).await?;
            Ok(result.prompts)
        }

        /// Invoke `prompts/get` with the given name and optional arguments map.
        ///
        /// Returns the expanded prompt messages that should be injected into the
        /// conversation as-is (`prompts/get` MCP).
        pub async fn get_prompt(
            &self,
            name: &str,
            arguments: Option<std::collections::HashMap<String, String>>,
        ) -> anyhow::Result<GetPromptResult> {
            let mut params = serde_json::json!({ "name": name });
            if let Some(args) = arguments {
                params["arguments"] = serde_json::to_value(args)?;
            }
            let result: GetPromptResult = self.call("prompts/get", Some(params)).await?;
            Ok(result)
        }

        /// Get all tools as `ToolDefinition` objects suitable for the API.
        pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
            self.tools.iter().map(|t| t.into()).collect()
        }

        /// Access the transport for subscribing to notifications.
        pub fn transport(&self) -> Arc<dyn transport::McpTransport> {
            Arc::clone(&self.transport)
        }

        // ---- Notification dispatch ----------------------------------------

        /// Drain any pending server-initiated notifications from the transport
        /// and route them to the appropriate subscribers in `resource_subscriptions`.
        ///
        /// Only messages that have a `"method"` field but no non-null `"id"` field
        /// are treated as notifications; everything else is skipped (this method
        /// does NOT consume RPC response messages).
        ///
        /// Handled notification methods:
        /// - `notifications/resources/updated` — delivers a [`ResourceChangedEvent`]
        ///   to the matching sender in `resource_subscriptions`.
        /// - `notifications/tools/list_changed` — logged at info level.
        /// - anything else — logged at debug level.
        pub(crate) async fn poll_notifications(
            &self,
            resource_subscriptions: &dashmap::DashMap<
                (String, String),
                tokio::sync::mpsc::Sender<ResourceChangedEvent>,
            >,
        ) {
            let mut notification_rx = self.notification_rx.lock().await;
            while let Some(raw) = notification_rx.recv().await {
                self.process_notification(raw, resource_subscriptions).await;
            }
        }

        /// Process a single notification message from the transport stream.
        /// Routes resource updates to subscribers and logs other notifications.
        pub(crate) async fn process_notification(
            &self,
            raw: serde_json::Value,
            resource_subscriptions: &dashmap::DashMap<
                (String, String),
                tokio::sync::mpsc::Sender<ResourceChangedEvent>,
            >,
        ) {
            // Only process server-initiated notifications (have "method", no non-null "id")
            let has_method = raw.get("method").is_some();
            let has_id = raw.get("id").map(|v| !v.is_null()).unwrap_or(false);
            if !has_method || has_id {
                // This is an RPC response, not a notification — skip it.
                debug!(
                    server = %self.server_name,
                    "process_notification: skipping non-notification message"
                );
                return;
            }

            let method = raw["method"].as_str().unwrap_or("");
            match method {
                "notifications/resources/updated" => {
                    let uri = raw["params"]["uri"].as_str().unwrap_or("").to_string();
                    let key = (self.server_name.clone(), uri.clone());
                    if let Some(tx) = resource_subscriptions.get(&key) {
                        let event = ResourceChangedEvent {
                            server_name: self.server_name.clone(),
                            uri,
                        };
                        if let Err(e) = tx.send(event).await {
                            debug!(
                                server = %self.server_name,
                                error = %e,
                                "process_notification: resource subscription receiver dropped"
                            );
                        }
                    } else {
                        debug!(
                            server = %self.server_name,
                            uri = %raw["params"]["uri"],
                            "process_notification: no subscriber for resource update"
                        );
                    }
                }
                "notifications/tools/list_changed" => {
                    info!(server = %self.server_name, "MCP tools list changed");
                }
                other => {
                    debug!(
                        server = %self.server_name,
                        method = %other,
                        "Unhandled MCP notification"
                    );
                }
            }
        }

        // ---- Internal RPC machinery ---------------------------------------

        /// Send a request and wait for the response, deserializing into T.
        pub(crate) async fn call<T: for<'de> Deserialize<'de>>(
            &self,
            method: &str,
            params: Option<Value>,
        ) -> anyhow::Result<T> {
            let (_id, resp_rx) = self.send_request(method, params).await?;
            let resp = resp_rx.await.map_err(|_| {
                anyhow::anyhow!(
                    "MCP transport closed while waiting for response to '{}'",
                    method
                )
            })?;

            if let Some(err) = resp.error {
                return Err(anyhow::anyhow!(
                    "MCP error {} from '{}': {}",
                    err.code,
                    method,
                    err.message
                ));
            }

            let result = resp
                .result
                .ok_or_else(|| anyhow::anyhow!("No result in MCP response for '{}'", method))?;
            Ok(serde_json::from_value(result)?)
        }

        async fn send_request(
            &self,
            method: &str,
            params: Option<Value>,
        ) -> anyhow::Result<(u64, oneshot::Receiver<JsonRpcResponse>)> {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let req = JsonRpcRequest::new(id, method, params);
            let (tx, rx) = oneshot::channel();
            self.pending.lock().await.insert(id, tx);
            if let Err(e) = self.transport.send(&req).await {
                let _ = self.pending.lock().await.remove(&id);
                return Err(e);
            }
            Ok((id, rx))
        }

        /// Test-only constructor: build an `McpClient` backed by an arbitrary
        /// transport without going through the real MCP handshake.
        #[cfg(test)]
        pub fn new_for_test(
            server_name: impl Into<String>,
            transport: Arc<dyn transport::McpTransport>,
        ) -> Self {
            let pending = Arc::new(Mutex::new(HashMap::new()));
            let (notification_tx, notification_rx) = mpsc::unbounded_channel();
            let server_name = server_name.into();

            Self::start_message_router(
                server_name.clone(),
                Arc::clone(&transport),
                Arc::clone(&pending),
                notification_tx,
            );

            Self {
                server_name,
                server_info: None,
                capabilities: ServerCapabilities::default(),
                tools: vec![],
                resources: vec![],
                prompts: vec![],
                instructions: None,
                transport,
                next_id: AtomicU64::new(1),
                pending,
                notification_rx: Arc::new(Mutex::new(notification_rx)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Auth State
// ---------------------------------------------------------------------------

/// Authentication state for a single MCP server.
#[derive(Debug, Clone)]
pub enum McpAuthState {
    /// Server does not require OAuth authentication.
    NotRequired,
    /// OAuth required; `auth_url` is where the user should go.
    Required { auth_url: String },
    /// Successfully authenticated; token may have an expiry.
    Authenticated {
        token_expiry: Option<chrono::DateTime<chrono::Utc>>,
    },
    /// An error occurred reading / initiating auth.
    Error(String),
}

// ---------------------------------------------------------------------------
// MCP Manager: manages multiple server connections
// ---------------------------------------------------------------------------

/// Manages a pool of MCP server connections.
pub struct McpManager {
    clients: HashMap<String, Arc<McpClient>>,
    /// Servers that failed to connect during `connect_all`.
    failed_servers: Vec<(String, String)>, // (name, error)
    /// Original (unexpanded) server configs — needed for OAuth initiation.
    server_configs: HashMap<String, McpServerConfig>,
    /// Active resource subscriptions: (server_name, uri) → change event sender.
    pub resource_subscriptions:
        DashMap<(String, String), tokio::sync::mpsc::Sender<ResourceChangedEvent>>,
}

#[derive(Debug, Clone)]
pub struct McpServerCatalog {
    pub tool_count: usize,
    pub resource_count: usize,
    pub prompt_count: usize,
    pub resources: Vec<String>,
    pub prompts: Vec<String>,
}

impl McpManager {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            failed_servers: Vec::new(),
            server_configs: HashMap::new(),
            resource_subscriptions: DashMap::new(),
        }
    }

    /// Connect to all configured MCP servers.
    ///
    /// - Expands env vars in each config before connecting.
    /// - Logs success/failure clearly.
    /// - Continues on failure (does not bail out on first error).
    /// - Tracks failed servers in `failed_servers()`.
    pub async fn connect_all(configs: &[McpServerConfig]) -> Self {
        let mut manager = Self::new();
        for config in configs {
            // Store original config for later OAuth use
            manager
                .server_configs
                .insert(config.name.clone(), config.clone());
            // Expand env vars before using the config
            let expanded = expand_server_config(config);

            match expanded.server_type.as_str() {
                "stdio" => {
                    debug!(
                        server = %expanded.name,
                        command = ?expanded.command,
                        "Connecting to MCP server via stdio"
                    );
                    match McpClient::connect_stdio(&expanded).await {
                        Ok(client) => {
                            info!(
                                server = %expanded.name,
                                tools = client.tools.len(),
                                resources = client.resources.len(),
                                "MCP server connected"
                            );
                            manager
                                .clients
                                .insert(expanded.name.clone(), Arc::new(client));
                        }
                        Err(e) => {
                            error!(
                                server = %expanded.name,
                                error = %e,
                                "Failed to connect to MCP server"
                            );
                            manager
                                .failed_servers
                                .push((expanded.name.clone(), e.to_string()));
                        }
                    }
                }
                "http" | "sse" | "pipedream" => {
                    debug!(
                        server = %expanded.name,
                        url = ?expanded.url,
                        transport = %expanded.server_type,
                        "Connecting to MCP server via remote transport"
                    );
                    match McpClient::connect_remote(&expanded).await {
                        Ok(client) => {
                            info!(
                                server = %expanded.name,
                                tools = client.tools.len(),
                                resources = client.resources.len(),
                                "MCP remote server connected"
                            );
                            manager
                                .clients
                                .insert(expanded.name.clone(), Arc::new(client));
                        }
                        Err(e) => {
                            error!(
                                server = %expanded.name,
                                error = %e,
                                "Failed to connect to remote MCP server"
                            );
                            manager
                                .failed_servers
                                .push((expanded.name.clone(), e.to_string()));
                        }
                    }
                }
                other => {
                    warn!(
                        server = %expanded.name,
                        transport = other,
                        "Unsupported MCP transport type; skipping server"
                    );
                    manager.failed_servers.push((
                        expanded.name.clone(),
                        format!("unsupported transport: {}", other),
                    ));
                }
            }
        }
        manager
    }

    // -----------------------------------------------------------------------
    // Status / query API (used by /mcp command and McpConnectionManager)
    // -----------------------------------------------------------------------

    /// Return all connected server names.
    pub fn server_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Return status for a single server.
    pub fn server_status(&self, name: &str) -> McpServerStatus {
        if let Some(client) = self.clients.get(name) {
            McpServerStatus::Connected {
                tool_count: client.tools.len(),
            }
        } else if let Some((_, err)) = self.failed_servers.iter().find(|(n, _)| n == name) {
            McpServerStatus::Disconnected {
                last_error: Some(err.clone()),
            }
        } else {
            McpServerStatus::Disconnected { last_error: None }
        }
    }

    /// Return status for every configured server (connected + failed).
    pub fn all_statuses(&self) -> HashMap<String, McpServerStatus> {
        let mut map = HashMap::new();
        for (name, client) in &self.clients {
            map.insert(
                name.clone(),
                McpServerStatus::Connected {
                    tool_count: client.tools.len(),
                },
            );
        }
        for (name, err) in &self.failed_servers {
            map.insert(
                name.clone(),
                McpServerStatus::Disconnected {
                    last_error: Some(err.clone()),
                },
            );
        }
        map
    }

    /// Servers that failed to connect during `connect_all`.
    /// Each entry is `(server_name, error_message)`.
    pub fn failed_servers(&self) -> &[(String, String)] {
        &self.failed_servers
    }

    /// Return counts and names for tools/resources/prompts on connected servers.
    pub fn server_catalog(&self, name: &str) -> Option<McpServerCatalog> {
        let client = self.clients.get(name)?;
        Some(McpServerCatalog {
            tool_count: client.tools.len(),
            resource_count: client.resources.len(),
            prompt_count: client.prompts.len(),
            resources: client.resources.iter().map(|r| r.name.clone()).collect(),
            prompts: client.prompts.iter().map(|p| p.name.clone()).collect(),
        })
    }

    // -----------------------------------------------------------------------
    // Tool / resource helpers
    // -----------------------------------------------------------------------

    /// Get all tool definitions from all connected servers.
    pub fn all_tool_definitions(&self) -> Vec<(String, ToolDefinition)> {
        let mut defs = vec![];
        for (server_name, client) in &self.clients {
            for td in client.tool_definitions() {
                // Prefix tool name with server name to avoid conflicts
                let prefixed = ToolDefinition {
                    name: format!("{}_{}", server_name, td.name),
                    description: format!("[{}] {}", server_name, td.description),
                    input_schema: td.input_schema.clone(),
                };
                defs.push((server_name.clone(), prefixed));
            }
        }
        defs
    }

    /// Execute a tool call, routing to the correct server.
    /// Tool name format: `<server_name>_<tool_name>`.
    pub async fn call_tool(
        &self,
        prefixed_name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<CallToolResult> {
        // Find the server name by matching prefix
        for (server_name, client) in &self.clients {
            let prefix = format!("{}_", server_name);
            if let Some(tool_name) = prefixed_name.strip_prefix(&prefix) {
                return client.call_tool(tool_name, arguments).await;
            }
        }
        Err(anyhow::anyhow!(
            "No MCP server found for tool '{}'. Connected servers: [{}]",
            prefixed_name,
            self.clients.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    }

    /// Number of connected servers.
    pub fn server_count(&self) -> usize {
        self.clients.len()
    }

    /// Get server instructions (from initialize response).
    pub fn server_instructions(&self) -> Vec<(String, String)> {
        self.clients
            .iter()
            .filter_map(|(name, client)| {
                client
                    .instructions
                    .as_ref()
                    .map(|instr| (name.clone(), instr.clone()))
            })
            .collect()
    }

    /// List all resources from all (or a specific) connected server.
    pub async fn list_all_resources(&self, server_filter: Option<&str>) -> Vec<serde_json::Value> {
        let mut all = vec![];
        for (name, client) in &self.clients {
            if let Some(filter) = server_filter {
                if name != filter {
                    continue;
                }
            }
            match client.list_resources().await {
                Ok(resources) => {
                    for r in resources {
                        all.push(serde_json::json!({
                            "uri": r.uri,
                            "name": r.name,
                            "description": r.description,
                            "mimeType": r.mime_type,
                            "server": name,
                        }));
                    }
                }
                Err(e) => {
                    warn!(server = %name, error = %e, "Failed to list resources");
                }
            }
        }
        all
    }

    /// Read a specific resource from a named server.
    pub async fn read_resource(
        &self,
        server_name: &str,
        uri: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let client = self.clients.get(server_name).ok_or_else(|| {
            anyhow::anyhow!("MCP server '{}' not found or not connected", server_name)
        })?;

        let contents = client.read_resource(uri).await?;
        Ok(serde_json::to_value(&contents)?)
    }

    /// List all prompts from all (or a specific) connected server.
    pub async fn list_all_prompts(&self, server_filter: Option<&str>) -> Vec<serde_json::Value> {
        let mut all = vec![];
        for (name, client) in &self.clients {
            if let Some(filter) = server_filter {
                if name != filter {
                    continue;
                }
            }
            match client.list_prompts().await {
                Ok(prompts) => {
                    for p in prompts {
                        all.push(serde_json::json!({
                            "name": p.name,
                            "description": p.description,
                            "arguments": p.arguments,
                            "server": name,
                        }));
                    }
                }
                Err(e) => {
                    warn!(server = %name, error = %e, "Failed to list prompts");
                }
            }
        }
        all
    }

    /// Get an expanded prompt from a named server by prompt name and arguments.
    ///
    /// Returns the `GetPromptResult` with fully-rendered messages suitable for
    /// injection into the conversation (`prompts/get` MCP).
    pub async fn get_prompt(
        &self,
        server_name: &str,
        prompt_name: &str,
        arguments: Option<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<GetPromptResult> {
        let client = self.clients.get(server_name).ok_or_else(|| {
            anyhow::anyhow!("MCP server '{}' not found or not connected", server_name)
        })?;
        client.get_prompt(prompt_name, arguments).await
    }

    // -----------------------------------------------------------------------
    // OAuth / authentication helpers
    // -----------------------------------------------------------------------

    /// Return the current authentication state for a server.
    ///
    /// - Returns `Authenticated` if a valid (non-expired) token exists on disk.
    /// - Returns `NotRequired` for stdio servers (they don't use OAuth).
    /// - Returns `Required` for HTTP servers that lack a valid token.
    pub fn auth_state(&self, server_name: &str) -> McpAuthState {
        // Check whether a token is already stored
        if let Some(token) = oauth::get_mcp_token(server_name) {
            if !token.is_expired(60) {
                let token_expiry = token.expires_at.map(|ts| {
                    chrono::DateTime::<chrono::Utc>::from(
                        std::time::UNIX_EPOCH + std::time::Duration::from_secs(ts),
                    )
                });
                return McpAuthState::Authenticated { token_expiry };
            }
        }

        // Determine server type from stored configs
        let config = match self.server_configs.get(server_name) {
            Some(c) => c,
            None => return McpAuthState::NotRequired,
        };

        let has_auth_headers = config.headers.keys().any(|key| {
            key.eq_ignore_ascii_case("authorization") || key.eq_ignore_ascii_case("x-api-key")
        });
        if has_auth_headers {
            return McpAuthState::NotRequired;
        }

        match config.server_type.as_str() {
            "pipedream" => McpAuthState::NotRequired,
            "http" | "sse" => McpAuthState::Required {
                auth_url: config
                    .url
                    .clone()
                    .unwrap_or_else(|| "(unknown URL)".to_string()),
            },
            _ => McpAuthState::NotRequired,
        }
    }

    /// Initiate OAuth 2.0 + PKCE for an HTTP MCP server.
    ///
    /// 1. GETs `<server_url>/.well-known/oauth-authorization-server`
    /// 2. Parses `authorization_endpoint`
    /// 3. Generates PKCE challenge
    /// 4. Returns the full auth URL (browser opening done at the command layer)
    ///
    /// The PKCE verifier is *not* persisted here; it is embedded in the URL
    /// so the command layer can display it.  A full end-to-end exchange would
    /// store the verifier and wait for the callback — that is handled by
    /// `oauth::exchange_code` once the code is received.
    pub async fn initiate_auth(&self, server_name: &str) -> anyhow::Result<String> {
        let config = self
            .server_configs
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown MCP server: {}", server_name))?;

        let base_url = config
            .url
            .as_deref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MCP server '{}' has no URL configured (required for OAuth)",
                    server_name
                )
            })?
            .trim_end_matches('/');

        // 1. Fetch OAuth Authorization Server Metadata (RFC 8414)
        let metadata_url = format!("{}/.well-known/oauth-authorization-server", base_url);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;

        let authorization_endpoint = match client.get(&metadata_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let meta: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| anyhow::anyhow!("OAuth metadata parse error: {}", e))?;
                meta.get("authorization_endpoint")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "OAuth metadata for '{}' missing 'authorization_endpoint'",
                            server_name
                        )
                    })?
            }
            Ok(resp) => {
                // Metadata endpoint not found — fall back to <base_url>/oauth/authorize
                let status = resp.status();
                debug!(
                    server = %server_name,
                    status = %status,
                    "OAuth metadata endpoint returned non-success; using fallback"
                );
                format!("{}/oauth/authorize", base_url)
            }
            Err(e) => {
                // Network error — fall back
                debug!(server = %server_name, error = %e, "Failed to fetch OAuth metadata; using fallback");
                format!("{}/oauth/authorize", base_url)
            }
        };

        // 2. Allocate a redirect port
        let redirect_port = oauth::oauth_port_alloc()
            .map_err(|e| anyhow::anyhow!("Failed to allocate OAuth redirect port: {}", e))?;
        let redirect_uri = format!("http://127.0.0.1:{}/callback", redirect_port);

        // 3. Generate PKCE
        let verifier = oauth::pkce_verifier();
        let challenge = oauth::pkce_challenge(&verifier);

        // 4. Build auth URL
        let auth_url = format!(
            "{}?client_id=mangocode&redirect_uri={}&response_type=code&code_challenge={}&code_challenge_method=S256",
            authorization_endpoint,
            urlencoding::encode(&redirect_uri),
            challenge,
        );

        Ok(auth_url)
    }

    /// Store an OAuth access token for an MCP server.
    ///
    /// `expires_in` is the lifetime in seconds (as returned by the token endpoint).
    pub fn store_token(
        &self,
        server_name: &str,
        token: &str,
        expires_in: Option<u64>,
    ) -> anyhow::Result<()> {
        let expires_at = expires_in.map(|secs| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + secs
        });
        let mcp_token = oauth::McpToken {
            access_token: token.to_string(),
            refresh_token: None,
            expires_at,
            scope: None,
            server_name: server_name.to_string(),
        };
        oauth::store_mcp_token(&mcp_token)
            .map_err(|e| anyhow::anyhow!("Failed to store MCP token for '{}': {}", server_name, e))
    }

    /// Load the stored OAuth access token for an MCP server, if any.
    ///
    /// Returns `None` if no token is stored or the token is expired.
    pub fn load_token(&self, server_name: &str) -> Option<String> {
        let token = oauth::get_mcp_token(server_name)?;
        if token.is_expired(60) {
            None
        } else {
            Some(token.access_token)
        }
    }

    // -----------------------------------------------------------------------
    // Notification dispatch loop
    // -----------------------------------------------------------------------

    /// Spawn background Tokio tasks for each connected MCP client to handle
    /// server-initiated notifications via async streams. Uses native push notifications
    /// when available (e.g., WebSocket) and falls back to polling for other transports (e.g., stdio).
    ///
    /// Routes `notifications/resources/updated` events to the appropriate sender in
    /// `self.resource_subscriptions`.
    ///
    /// Call this once after constructing an `Arc<McpManager>` (e.g. immediately
    /// after `McpManager::connect_all`).  Each notification handler task exits
    /// when the transport closes or the manager is dropped.
    pub fn spawn_notification_poll_loop(self: Arc<Self>) {
        let clients = self.clients.clone();

        // Spawn a task for each client to handle notifications via the stream
        for client in clients.values() {
            let client_clone = Arc::clone(client);
            let manager_weak = Arc::downgrade(&self);

            tokio::spawn(async move {
                let manager = match manager_weak.upgrade() {
                    Some(m) => m,
                    None => return,
                };

                client_clone
                    .poll_notifications(&manager.resource_subscriptions)
                    .await;
            });
        }
    }
}

impl Default for McpManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MCP result → string conversion
// ---------------------------------------------------------------------------

/// Convert an MCP tool call result to a string for the model.
///
/// Content type handling:
/// - `text`     → the text itself
/// - `image`    → `[Image: <mime_type>]` with a short base64 preview
/// - `resource` → `[Resource: <uri>]` plus text content if present
///
/// Mixed content is joined with newlines.
/// If all content is empty, returns an empty string.
pub fn mcp_result_to_string(result: &CallToolResult) -> String {
    let parts: Vec<String> = result
        .content
        .iter()
        .map(|c| match c {
            McpContent::Text { text } => text.clone(),
            McpContent::Image { data, mime_type } => {
                // Show a short preview (first 32 chars of base64) so the model
                // knows an image was returned without embedding the full blob.
                let preview_len = data.len().min(32);
                let preview = &data[..preview_len];
                let ellipsis = if data.len() > 32 { "…" } else { "" };
                format!(
                    "[Image: {} | base64 preview: {}{}]",
                    mime_type, preview, ellipsis
                )
            }
            McpContent::Resource { resource } => {
                let mut parts = vec![format!("[Resource: {}]", resource.uri)];
                if let Some(ref text) = resource.text {
                    parts.push(text.clone());
                } else if resource.blob.is_some() {
                    let mime = resource
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    parts.push(format!("[Binary resource: {}]", mime));
                }
                parts.join("\n")
            }
        })
        .collect();

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- env expansion -----------------------------------------------------

    #[test]
    fn test_expand_env_vars_no_vars() {
        assert_eq!(expand_env_vars("hello world"), "hello world");
    }

    #[test]
    fn test_expand_env_vars_known_var() {
        std::env::set_var("_CC_TEST_VAR", "rustacean");
        let out = expand_env_vars("hello ${_CC_TEST_VAR}!");
        assert_eq!(out, "hello rustacean!");
        std::env::remove_var("_CC_TEST_VAR");
    }

    #[test]
    fn test_expand_env_vars_default_value() {
        std::env::remove_var("_CC_MISSING_VAR");
        let out = expand_env_vars("val=${_CC_MISSING_VAR:-fallback}");
        assert_eq!(out, "val=fallback");
    }

    #[test]
    fn test_expand_env_vars_missing_no_default() {
        std::env::remove_var("_CC_REALLY_MISSING");
        // Missing with no default → keep original
        let out = expand_env_vars("${_CC_REALLY_MISSING}");
        assert_eq!(out, "${_CC_REALLY_MISSING}");
    }

    #[test]
    fn test_expand_env_vars_multiple() {
        std::env::set_var("_CC_A", "foo");
        std::env::set_var("_CC_B", "bar");
        let out = expand_env_vars("${_CC_A}/${_CC_B}");
        assert_eq!(out, "foo/bar");
        std::env::remove_var("_CC_A");
        std::env::remove_var("_CC_B");
    }

    #[test]
    fn test_expand_server_config() {
        std::env::set_var("_CC_TEST_HOME", "/home/user");
        let cfg = McpServerConfig {
            name: "test".to_string(),
            command: Some("${_CC_TEST_HOME}/bin/server".to_string()),
            args: vec!["--root".to_string(), "${_CC_TEST_HOME}".to_string()],
            env: {
                let mut m = HashMap::new();
                m.insert("PATH".to_string(), "${_CC_TEST_HOME}/bin".to_string());
                m
            },
            url: None,
            headers: {
                let mut m = HashMap::new();
                m.insert(
                    "Authorization".to_string(),
                    "Bearer ${_CC_TEST_HOME}".to_string(),
                );
                m
            },
            pipedream: None,
            server_type: "stdio".to_string(),
        };
        let expanded = expand_server_config(&cfg);
        assert_eq!(expanded.command.as_deref(), Some("/home/user/bin/server"));
        assert_eq!(expanded.args[1], "/home/user");
        assert_eq!(
            expanded.env.get("PATH").map(|s| s.as_str()),
            Some("/home/user/bin")
        );
        assert_eq!(
            expanded.headers.get("Authorization").map(|s| s.as_str()),
            Some("Bearer /home/user")
        );
        std::env::remove_var("_CC_TEST_HOME");
    }

    // ---- JSON-RPC -----------------------------------------------------------

    #[test]
    fn test_json_rpc_request_serialization() {
        let req = JsonRpcRequest::new(1u64, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    // ---- McpTool → ToolDefinition ------------------------------------------

    #[test]
    fn test_mcp_tool_to_definition() {
        let tool = McpTool {
            name: "search".to_string(),
            description: Some("Search the web".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
        };
        let def: ToolDefinition = (&tool).into();
        assert_eq!(def.name, "search");
        assert_eq!(def.description, "Search the web");
    }

    // ---- mcp_result_to_string ----------------------------------------------

    #[test]
    fn test_result_to_string_text() {
        let result = CallToolResult {
            content: vec![McpContent::Text {
                text: "hello".to_string(),
            }],
            is_error: false,
        };
        assert_eq!(mcp_result_to_string(&result), "hello");
    }

    #[test]
    fn test_result_to_string_image() {
        let result = CallToolResult {
            content: vec![McpContent::Image {
                data: "abc123".to_string(),
                mime_type: "image/png".to_string(),
            }],
            is_error: false,
        };
        let s = mcp_result_to_string(&result);
        assert!(s.contains("Image:"));
        assert!(s.contains("image/png"));
        assert!(s.contains("abc123"));
    }

    #[test]
    fn test_result_to_string_resource_with_text() {
        let result = CallToolResult {
            content: vec![McpContent::Resource {
                resource: ResourceContents {
                    uri: "file:///foo.txt".to_string(),
                    mime_type: Some("text/plain".to_string()),
                    text: Some("file contents".to_string()),
                    blob: None,
                },
            }],
            is_error: false,
        };
        let s = mcp_result_to_string(&result);
        assert!(s.contains("[Resource: file:///foo.txt]"));
        assert!(s.contains("file contents"));
    }

    #[test]
    fn test_result_to_string_resource_binary() {
        let result = CallToolResult {
            content: vec![McpContent::Resource {
                resource: ResourceContents {
                    uri: "file:///img.png".to_string(),
                    mime_type: Some("image/png".to_string()),
                    text: None,
                    blob: Some("BASE64==".to_string()),
                },
            }],
            is_error: false,
        };
        let s = mcp_result_to_string(&result);
        assert!(s.contains("[Resource: file:///img.png]"));
        assert!(s.contains("[Binary resource: image/png]"));
    }

    #[test]
    fn test_result_to_string_mixed() {
        let result = CallToolResult {
            content: vec![
                McpContent::Text {
                    text: "line one".to_string(),
                },
                McpContent::Text {
                    text: "line two".to_string(),
                },
            ],
            is_error: false,
        };
        assert_eq!(mcp_result_to_string(&result), "line one\nline two");
    }

    // ---- McpManager --------------------------------------------------------

    #[test]
    fn test_manager_server_names_empty() {
        let mgr = McpManager::new();
        assert!(mgr.server_names().is_empty());
    }

    #[test]
    fn test_manager_all_statuses_empty() {
        let mgr = McpManager::new();
        assert!(mgr.all_statuses().is_empty());
    }

    #[test]
    fn test_manager_failed_servers_empty() {
        let mgr = McpManager::new();
        assert!(mgr.failed_servers().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Resource subscriptions (T2-12)
// ---------------------------------------------------------------------------

use tokio::sync::mpsc as tokio_mpsc;

/// Notification that a resource has changed.
#[derive(Debug, Clone)]
pub struct ResourceChangedEvent {
    pub server_name: String,
    pub uri: String,
}

/// Subscription handle for a single MCP resource URI.
pub struct ResourceSubscription {
    pub server_name: String,
    pub uri: String,
}

/// Subscribe to resource changes on an MCP server.
///
/// Sends the `resources/subscribe` JSON-RPC request to the named server and
/// returns a channel receiver that will deliver [`ResourceChangedEvent`] values
/// whenever the server fires a `notifications/resources/updated` notification.
/// The notification dispatch loop (elsewhere) looks up the tx in
/// `manager.resource_subscriptions` and forwards events.
///
/// If the server is not connected or the RPC fails, a dead receiver is returned
/// (no events will ever be delivered).
pub async fn subscribe_resource(
    manager: &McpManager,
    server_name: &str,
    uri: &str,
) -> tokio_mpsc::Receiver<ResourceChangedEvent> {
    let make_dead = || {
        let (_tx, rx) = tokio_mpsc::channel::<ResourceChangedEvent>(1);
        rx
    };

    let client = match manager.clients.get(server_name) {
        Some(c) => c,
        None => {
            tracing::warn!(server_name, uri, "subscribe_resource: server not connected");
            return make_dead();
        }
    };

    let params = serde_json::json!({ "uri": uri });
    if let Err(e) = client
        .call::<serde_json::Value>("resources/subscribe", Some(params))
        .await
    {
        tracing::warn!(server_name, uri, error = %e, "subscribe_resource RPC failed");
        return make_dead();
    }

    let (tx, rx) = tokio_mpsc::channel(32);
    manager
        .resource_subscriptions
        .insert((server_name.to_string(), uri.to_string()), tx);
    tracing::info!(server_name, uri, "MCP resource subscription registered");
    rx
}

/// Unsubscribe from resource change notifications.
///
/// Sends `resources/unsubscribe` JSON-RPC request to the named server via
/// `McpManager`.  Returns an error if the server is not connected or the
/// request fails.
pub async fn unsubscribe_resource(
    manager: &McpManager,
    server_name: &str,
    uri: &str,
) -> Result<(), String> {
    let client = manager.clients.get(server_name).ok_or_else(|| {
        format!(
            "unsubscribe_resource: server '{}' not connected",
            server_name
        )
    })?;

    let params = serde_json::json!({ "uri": uri });
    client
        .call_tool("resources/unsubscribe", Some(params))
        .await
        .map_err(|e| format!("unsubscribe_resource failed: {e}"))
        .map(|_| ())
}

// ---------------------------------------------------------------------------
// Notification dispatch tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod notification_tests {
    use super::*;

    /// A mock transport that returns pre-queued raw JSON lines and discards sends.
    struct MockTransport {
        queue: tokio::sync::Mutex<std::collections::VecDeque<String>>,
    }

    impl MockTransport {
        fn with_lines(lines: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                queue: tokio::sync::Mutex::new(lines.iter().map(|s| s.to_string()).collect()),
            })
        }
    }

    #[async_trait::async_trait]
    impl transport::McpTransport for MockTransport {
        async fn send(&self, _msg: &JsonRpcRequest) -> anyhow::Result<()> {
            Ok(())
        }

        async fn recv(&self) -> anyhow::Result<Option<JsonRpcResponse>> {
            Ok(None)
        }

        async fn close(&self) -> anyhow::Result<()> {
            Ok(())
        }

        async fn set_protocol_version(&self, _version: Option<String>) -> anyhow::Result<()> {
            Ok(())
        }

        async fn try_receive_raw(&self) -> anyhow::Result<Option<serde_json::Value>> {
            let mut q = self.queue.lock().await;
            match q.pop_front() {
                Some(line) => {
                    let v: serde_json::Value = serde_json::from_str(&line)?;
                    Ok(Some(v))
                }
                None => Ok(None),
            }
        }

        fn subscribe_to_notifications(
            &self,
        ) -> BoxStream<'static, anyhow::Result<serde_json::Value>> {
            let snapshot = self
                .queue
                .try_lock()
                .map(|q| q.iter().cloned().collect::<std::collections::VecDeque<_>>())
                .unwrap_or_default();
            let queue = Arc::new(tokio::sync::Mutex::new(snapshot));

            let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<serde_json::Value>>(100);

            // Spawn a background task that yields queued notifications
            tokio::spawn(async move {
                loop {
                    let line = {
                        let mut q = queue.lock().await;
                        q.pop_front()
                    };

                    match line {
                        Some(s) => {
                            let val: anyhow::Result<serde_json::Value> = serde_json::from_str(&s)
                                .map_err(|e| {
                                    anyhow::anyhow!("Mock parse error: {} (raw: {})", e, s)
                                });

                            if tx.send(val).await.is_err() {
                                break;
                            }
                        }
                        None => {
                            // Queue exhausted
                            break;
                        }
                    }
                }
            });

            Box::pin(ReceiverStream::new(rx))
        }
    }

    #[tokio::test]
    async fn test_poll_notifications_routes_resource_updated() {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": { "uri": "file:///foo.txt" }
        })
        .to_string();

        let client = client::McpClient::new_for_test(
            "myserver",
            MockTransport::with_lines(&[&notification]),
        );

        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        let (tx, mut rx) = tokio_mpsc::channel::<ResourceChangedEvent>(4);
        subscriptions.insert(("myserver".to_string(), "file:///foo.txt".to_string()), tx);

        client.poll_notifications(&subscriptions).await;

        let event = rx.try_recv().expect("expected a ResourceChangedEvent");
        assert_eq!(event.server_name, "myserver");
        assert_eq!(event.uri, "file:///foo.txt");
        assert!(
            rx.try_recv().is_err(),
            "channel should be empty after one event"
        );
    }

    #[tokio::test]
    async fn test_poll_notifications_no_subscriber_does_not_panic() {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": { "uri": "file:///unsubscribed.txt" }
        })
        .to_string();

        let client = client::McpClient::new_for_test(
            "myserver",
            MockTransport::with_lines(&[&notification]),
        );
        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        // No subscriber registered — should silently skip without panicking.
        client.poll_notifications(&subscriptions).await;
    }

    #[tokio::test]
    async fn test_poll_notifications_tools_list_changed() {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
            "params": {}
        })
        .to_string();

        let client = client::McpClient::new_for_test(
            "myserver",
            MockTransport::with_lines(&[&notification]),
        );
        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        client.poll_notifications(&subscriptions).await; // must not panic
    }

    #[tokio::test]
    async fn test_poll_notifications_empty_queue_is_noop() {
        let client = client::McpClient::new_for_test("myserver", MockTransport::with_lines(&[]));
        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        // Must return immediately without blocking or panicking.
        client.poll_notifications(&subscriptions).await;
    }

    #[tokio::test]
    async fn test_poll_notifications_multiple_events() {
        let n1 = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": { "uri": "file:///a.txt" }
        })
        .to_string();
        let n2 = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": { "uri": "file:///b.txt" }
        })
        .to_string();

        let client = client::McpClient::new_for_test("s1", MockTransport::with_lines(&[&n1, &n2]));

        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        let (tx_a, mut rx_a) = tokio_mpsc::channel::<ResourceChangedEvent>(4);
        let (tx_b, mut rx_b) = tokio_mpsc::channel::<ResourceChangedEvent>(4);
        subscriptions.insert(("s1".to_string(), "file:///a.txt".to_string()), tx_a);
        subscriptions.insert(("s1".to_string(), "file:///b.txt".to_string()), tx_b);

        client.poll_notifications(&subscriptions).await;

        let ev_a = rx_a.try_recv().expect("expected event for a.txt");
        assert_eq!(ev_a.uri, "file:///a.txt");

        let ev_b = rx_b.try_recv().expect("expected event for b.txt");
        assert_eq!(ev_b.uri, "file:///b.txt");
    }

    #[tokio::test]
    async fn test_poll_notifications_skips_rpc_responses() {
        // A message with a non-null "id" field is an RPC response — must not
        // be dispatched as a notification even if it has a "method" field.
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "notifications/resources/updated",
            "params": { "uri": "file:///foo.txt" }
        })
        .to_string();

        let client =
            client::McpClient::new_for_test("myserver", MockTransport::with_lines(&[&response]));

        let subscriptions: DashMap<
            (String, String),
            tokio::sync::mpsc::Sender<ResourceChangedEvent>,
        > = DashMap::new();
        let (tx, mut rx) = tokio_mpsc::channel::<ResourceChangedEvent>(4);
        subscriptions.insert(("myserver".to_string(), "file:///foo.txt".to_string()), tx);

        client.poll_notifications(&subscriptions).await;

        // The event must NOT have been delivered because the message has an id.
        assert!(
            rx.try_recv().is_err(),
            "RPC response must not be dispatched as a notification"
        );
    }
}
