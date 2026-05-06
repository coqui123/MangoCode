//! Agent Client Protocol (ACP) server for MangoCode.
//!
//! JSON-RPC 2.0 over stdio. The server is local-only and runs the native
//! MangoCode query loop directly; it does not proxy to any hosted service.

use mangocode_core::config::{Config, PermissionMode, Settings};
use mangocode_core::cost::CostTracker;
use mangocode_core::file_history::FileHistory;
use mangocode_core::types::{ContentBlock, Message, MessageContent, Role};
use mangocode_core::{PermissionDecision, PermissionHandler, PermissionRequest};
use mangocode_tools::ToolContext;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

type OutboundTx = mpsc::UnboundedSender<Value>;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcResponseMsg {
    pub id: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcErrorIn>,
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcErrorIn {
    pub code: i32,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct AcpServerState {
    sessions: Mutex<HashMap<String, Arc<tokio::sync::Mutex<AcpSession>>>>,
    running: Mutex<HashMap<String, CancellationToken>>,
    pending_permissions: Mutex<HashMap<String, PendingPermission>>,
}

struct PendingPermission {
    session_id: String,
    tx: std_mpsc::Sender<PermissionDecision>,
}

struct AcpSession {
    session: mangocode_core::history::ConversationSession,
    cwd: PathBuf,
    model_override: Option<String>,
    permission_mode: Option<PermissionMode>,
}

// ---------------------------------------------------------------------------
// Server entry-point
// ---------------------------------------------------------------------------

pub async fn run_acp_server() -> anyhow::Result<()> {
    let state = Arc::new(AcpServerState::default());
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();

    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(value) = out_rx.recv().await {
            if write_line(&mut stdout, &value).await.is_err() {
                break;
            }
        }
    });

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            let _ = out_tx.send(json!(JsonRpcResponse::error(
                None,
                -32700,
                "Parse error".to_string()
            )));
            continue;
        };

        if value.get("method").is_some() {
            match serde_json::from_value::<JsonRpcRequest>(value) {
                Ok(req) => {
                    let state = state.clone();
                    let out_tx = out_tx.clone();
                    tokio::spawn(async move {
                        dispatch_request(req, state, out_tx).await;
                    });
                }
                Err(e) => {
                    let _ = out_tx.send(json!(JsonRpcResponse::error(
                        None,
                        -32600,
                        format!("Invalid request: {}", e)
                    )));
                }
            }
        } else if value.get("id").is_some() {
            if let Ok(resp) = serde_json::from_value::<JsonRpcResponseMsg>(value) {
                handle_client_response(resp, &state);
            }
        }
    }

    drop(out_tx);
    let _ = writer.await;
    Ok(())
}

async fn write_line(stdout: &mut tokio::io::Stdout, value: &Value) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

async fn dispatch_request(req: JsonRpcRequest, state: Arc<AcpServerState>, out_tx: OutboundTx) {
    let id = req.id.clone();
    debug!(method = %req.method, "ACP dispatch");
    let response = match handle_request(req, state, out_tx.clone()).await {
        Ok(value) => JsonRpcResponse::success(id.clone(), value),
        Err(err) => JsonRpcResponse::error(id.clone(), -32000, err.to_string()),
    };
    if id.is_some() {
        let _ = out_tx.send(json!(response));
    }
}

async fn handle_request(
    req: JsonRpcRequest,
    state: Arc<AcpServerState>,
    out_tx: OutboundTx,
) -> anyhow::Result<Value> {
    match req.method.as_str() {
        "initialize" => Ok(initialize_result(req.params)),
        "authenticate" => Ok(json!({ "authenticated": true })),
        "tool/list" => tool_list_json(req.params, state).await,
        "model/list" => Ok(model_list_json()),
        "session/list" => Ok(json!({ "sessions": list_sessions_json().await })),
        "session/new" | "session/create" => create_session(req.params, state).await,
        "session/load" => load_session(req.params, state, out_tx).await,
        "session/prompt" | "session/message" => prompt_session(req.params, state, out_tx).await,
        "session/cancel" => cancel_session(req.params, state),
        "session/close" => close_session(req.params, state),
        "session/set_model" => set_session_model(req.params, state).await,
        "session/set_mode" => set_session_mode(req.params, state).await,
        other => Err(anyhow::anyhow!("Method not found: {}", other)),
    }
}

fn initialize_result(params: Option<Value>) -> Value {
    let protocol_version = params
        .as_ref()
        .and_then(|params| {
            params
                .get("protocolVersion")
                .or_else(|| params.get("protocol_version"))
                .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
        })
        .unwrap_or(1);

    json!({
        "protocolVersion": protocol_version,
        "protocol_version": protocol_version,
        "agentCapabilities": {
            "loadSession": true,
            "promptCapabilities": {
                "image": false,
                "audio": false,
                "embeddedContext": true
            },
            "mcpCapabilities": {
                "http": false,
                "sse": false
            }
        },
        "agentInfo": {
            "name": "mangocode",
            "title": "MangoCode",
            "version": env!("CARGO_PKG_VERSION")
        },
        "authMethods": [],
        "serverInfo": {
            "name": "mangocode",
            "title": "MangoCode",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "sessions": { "create": true, "list": true, "load": true, "cancel": true },
            "tools": { "list": true },
            "models": { "list": true, "set": true },
            "streaming": true,
            "permissions": true
        }
    })
}

// ---------------------------------------------------------------------------
// Session RPC
// ---------------------------------------------------------------------------

async fn create_session(
    params: Option<Value>,
    state: Arc<AcpServerState>,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params
        .get("sessionId")
        .or_else(|| params.get("session_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let cwd = params
        .get("cwd")
        .or_else(|| params.get("workingDirectory"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let model_override = params
        .get("model")
        .or_else(|| params.get("modelId"))
        .or_else(|| params.get("model_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let model = model_override
        .clone()
        .unwrap_or_else(|| mangocode_core::constants::DEFAULT_MODEL.to_string());
    let permission_mode = params
        .get("modeId")
        .or_else(|| params.get("mode_id"))
        .or_else(|| params.get("mode"))
        .and_then(Value::as_str)
        .and_then(parse_permission_mode);
    let modes = modes_json(permission_mode.as_ref().unwrap_or(&PermissionMode::Default));

    let mut session = mangocode_core::history::ConversationSession::new(model);
    session.id = session_id.clone();
    session.working_dir = Some(cwd.display().to_string());
    let acp_session = AcpSession {
        session,
        cwd,
        model_override,
        permission_mode,
    };
    state.sessions.lock().unwrap().insert(
        session_id.clone(),
        Arc::new(tokio::sync::Mutex::new(acp_session)),
    );

    Ok(json!({
        "sessionId": session_id,
        "session_id": session_id,
        "modes": modes
    }))
}

async fn load_session(
    params: Option<Value>,
    state: Arc<AcpServerState>,
    out_tx: OutboundTx,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/load requires sessionId"))?;
    let session = mangocode_core::history::load_session(&session_id).await?;
    let cwd = session
        .working_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let acp_session = AcpSession {
        model_override: Some(session.model.clone()),
        session,
        cwd,
        permission_mode: None,
    };
    let session_arc = Arc::new(tokio::sync::Mutex::new(acp_session));
    state
        .sessions
        .lock()
        .unwrap()
        .insert(session_id.clone(), session_arc.clone());

    let session_guard = session_arc.lock().await;
    for message in &session_guard.session.messages {
        send_session_update(
            &out_tx,
            &session_id,
            json!({
                "sessionUpdate": "message",
                "role": role_name(message),
                "content": message_text(message),
            }),
        );
    }

    Ok(json!({
        "sessionId": session_id,
        "session_id": session_id,
        "messageCount": session_guard.session.messages.len()
    }))
}

async fn prompt_session(
    params: Option<Value>,
    state: Arc<AcpServerState>,
    out_tx: OutboundTx,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/prompt requires sessionId"))?;
    let prompt = extract_prompt_text(&params)
        .ok_or_else(|| anyhow::anyhow!("session/prompt requires prompt text"))?;

    let session_arc = {
        let sessions = state.sessions.lock().unwrap();
        sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Unknown session: {}", session_id))?
    };
    let (cwd, model_override, mode_override) = {
        let session = session_arc.lock().await;
        (
            session.cwd.clone(),
            session.model_override.clone(),
            session.permission_mode.clone(),
        )
    };
    let runtime = build_runtime(
        &session_id,
        &cwd,
        model_override,
        mode_override,
        out_tx.clone(),
        state.clone(),
    )
    .await?;

    let cancel = CancellationToken::new();
    {
        let mut running = state.running.lock().unwrap();
        if running.contains_key(&session_id) {
            return Err(anyhow::anyhow!(
                "Session is already running: {}",
                session_id
            ));
        }
        running.insert(session_id.clone(), cancel.clone());
    }

    let mut harness_rx = mangocode_core::harness::subscribe_events();
    let update_session_id = session_id.clone();
    let update_tx = out_tx.clone();
    let update_task = tokio::spawn(async move {
        while let Ok(event) = harness_rx.recv().await {
            if event.session_id == update_session_id {
                if let Some(update) = acp_update_from_harness_event(&event) {
                    send_session_update(&update_tx, &update_session_id, update);
                }
            }
        }
    });

    let outcome = {
        let mut session_guard = session_arc.lock().await;
        session_guard.session.messages.push(Message::user(prompt));

        let outcome = mangocode_query::run_query_loop(
            runtime.client.as_ref(),
            &mut session_guard.session.messages,
            runtime.tools.as_slice(),
            &runtime.tool_ctx,
            &runtime.query_config,
            runtime.cost_tracker.clone(),
            None,
            cancel.clone(),
            None,
        )
        .await;

        session_guard.session.model = runtime.query_config.model.clone();
        session_guard.session.working_dir = Some(session_guard.cwd.display().to_string());
        session_guard.session.total_cost = runtime.cost_tracker.total_cost_usd();
        session_guard.session.total_tokens =
            runtime.cost_tracker.input_tokens() + runtime.cost_tracker.output_tokens();
        session_guard.session.updated_at = chrono::Utc::now();
        let _ = mangocode_core::history::save_session(&session_guard.session).await;
        outcome
    };

    state.running.lock().unwrap().remove(&session_id);
    update_task.abort();

    let outcome_json = match outcome {
        mangocode_query::QueryOutcome::EndTurn { message, usage } => json!({
            "status": "completed",
            "content": message.get_all_text(),
            "usage": usage,
        }),
        mangocode_query::QueryOutcome::MaxTokens {
            partial_message,
            usage,
        } => json!({
            "status": "max_tokens",
            "content": partial_message.get_all_text(),
            "usage": usage,
        }),
        mangocode_query::QueryOutcome::Cancelled => json!({ "status": "cancelled" }),
        mangocode_query::QueryOutcome::BudgetExceeded {
            cost_usd,
            limit_usd,
        } => json!({
            "status": "budget_exceeded",
            "cost_usd": cost_usd,
            "limit_usd": limit_usd,
        }),
        mangocode_query::QueryOutcome::Error(err) => json!({
            "status": "error",
            "error": err.to_string(),
        }),
    };
    send_session_update(
        &out_tx,
        &session_id,
        json!({
            "sessionUpdate": "status",
            "status": outcome_json
                .get("status")
                .cloned()
                .unwrap_or(Value::String("completed".to_string())),
            "outcome": outcome_json.clone(),
        }),
    );

    Ok(json!({
        "sessionId": session_id,
        "session_id": session_id,
        "outcome": outcome_json
    }))
}

fn cancel_session(params: Option<Value>, state: Arc<AcpServerState>) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/cancel requires sessionId"))?;
    if let Some(token) = state.running.lock().unwrap().remove(&session_id) {
        token.cancel();
    }
    cancel_pending_permissions(&state, &session_id);
    Ok(json!({ "sessionId": session_id, "cancelled": true }))
}

fn close_session(params: Option<Value>, state: Arc<AcpServerState>) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/close requires sessionId"))?;
    if let Some(token) = state.running.lock().unwrap().remove(&session_id) {
        token.cancel();
    }
    cancel_pending_permissions(&state, &session_id);
    state.sessions.lock().unwrap().remove(&session_id);
    Ok(json!({ "sessionId": session_id, "closed": true }))
}

async fn set_session_model(
    params: Option<Value>,
    state: Arc<AcpServerState>,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/set_model requires sessionId"))?;
    let model = params
        .get("modelId")
        .or_else(|| params.get("model_id"))
        .or_else(|| params.get("model"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("session/set_model requires modelId"))?
        .to_string();

    let session_arc = {
        let sessions = state.sessions.lock().unwrap();
        sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Unknown session: {}", session_id))?
    };
    {
        let mut session = session_arc.lock().await;
        session.session.model = model.clone();
        session.model_override = Some(model.clone());
    }
    Ok(json!({
        "sessionId": session_id,
        "session_id": session_id,
        "modelId": model,
        "model_id": model
    }))
}

async fn set_session_mode(
    params: Option<Value>,
    state: Arc<AcpServerState>,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let session_id = params_session_id(&params)
        .ok_or_else(|| anyhow::anyhow!("session/set_mode requires sessionId"))?;
    let mode_id = params
        .get("modeId")
        .or_else(|| params.get("mode_id"))
        .or_else(|| params.get("mode"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    let mode = parse_permission_mode(mode_id)
        .ok_or_else(|| anyhow::anyhow!("Unknown session mode: {}", mode_id))?;
    let session = {
        let sessions = state.sessions.lock().unwrap();
        sessions
            .get(&session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Unknown session: {}", session_id))?
    };
    session.lock().await.permission_mode = Some(mode.clone());
    Ok(json!({
        "sessionId": session_id,
        "session_id": session_id,
        "modeId": permission_mode_id(&mode),
        "mode_id": permission_mode_id(&mode),
        "modes": modes_json(&mode)
    }))
}

// ---------------------------------------------------------------------------
// Runtime construction
// ---------------------------------------------------------------------------

struct QueryRuntime {
    client: Arc<mangocode_api::AnthropicClient>,
    tools: Vec<Box<dyn mangocode_tools::Tool>>,
    tool_ctx: ToolContext,
    query_config: mangocode_query::QueryConfig,
    cost_tracker: Arc<CostTracker>,
}

async fn build_runtime(
    session_id: &str,
    cwd: &Path,
    model_override: Option<String>,
    mode_override: Option<PermissionMode>,
    out_tx: OutboundTx,
    state: Arc<AcpServerState>,
) -> anyhow::Result<QueryRuntime> {
    let mut config = acp_effective_config(cwd, model_override, mode_override, true).await;

    let model_registry = mangocode_api::ModelRegistry::new();
    if config.model.is_none() {
        config.model = Some(mangocode_api::effective_model_for_config(
            &config,
            &model_registry,
        ));
    }

    let (api_key, use_bearer_auth) = match config.resolve_auth_async().await {
        Some(auth) => auth,
        None if config.provider.as_deref().unwrap_or("anthropic") != "anthropic" => {
            (String::new(), false)
        }
        None => {
            return Err(anyhow::anyhow!(
                "No API key found. Configure a local provider or run MangoCode auth first."
            ));
        }
    };

    let client_config = mangocode_api::client::ClientConfig {
        api_key,
        api_base: config.resolve_api_base(),
        use_bearer_auth,
        ..Default::default()
    };
    let client = Arc::new(mangocode_api::AnthropicClient::new(client_config.clone())?);
    let provider_registry =
        Arc::new(mangocode_api::ProviderRegistry::from_environment_with_auth_store(client_config));

    let cost_tracker = CostTracker::new();
    let current_turn = Arc::new(AtomicUsize::new(0));
    let mcp_manager = connect_acp_mcp_manager(&config).await;
    let permission_handler: Arc<dyn PermissionHandler> = Arc::new(AcpPermissionHandler {
        session_id: session_id.to_string(),
        mode: config.permission_mode.clone(),
        out_tx,
        state,
    });
    let tool_ctx = ToolContext {
        working_dir: cwd.to_path_buf(),
        permission_mode: config.permission_mode.clone(),
        permission_handler,
        cost_tracker: cost_tracker.clone(),
        session_metrics: Some(mangocode_core::analytics::SessionMetrics::new()),
        session_id: session_id.to_string(),
        file_history: Arc::new(ParkingMutex::new(FileHistory::new())),
        current_turn,
        non_interactive: false,
        mcp_manager: mcp_manager.clone(),
        config: config.clone(),
    };

    #[cfg(any(feature = "tool-team-create", feature = "tool-team-delete"))]
    {
        static SWARM_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        SWARM_INIT.get_or_init(mangocode_query::init_team_swarm_runner);
    }
    let tools = build_acp_tools(mcp_manager, &config);

    let mut query_config =
        mangocode_query::QueryConfig::from_config_with_registry(&config, &model_registry);
    query_config.provider_registry = Some(provider_registry);
    query_config.model_registry = Some(Arc::new(model_registry));
    query_config.working_directory = Some(cwd.display().to_string());

    Ok(QueryRuntime {
        client,
        tools,
        tool_ctx,
        query_config,
        cost_tracker,
    })
}

async fn acp_effective_config(
    cwd: &Path,
    model_override: Option<String>,
    mode_override: Option<PermissionMode>,
    install_plugin_globals: bool,
) -> Config {
    let settings = Settings::load_hierarchical(cwd).await;
    let mut config = settings.effective_config();
    config.project_dir = Some(cwd.to_path_buf());
    if let Some(model) = model_override.filter(|model| !model.trim().is_empty()) {
        config.model = Some(model);
    }
    if let Some(mode) = mode_override {
        config.permission_mode = mode;
    }

    let plugin_registry = mangocode_plugins::load_plugins(cwd, &[]).await;
    if install_plugin_globals {
        mangocode_plugins::set_global_hooks(plugin_registry.build_hook_registry());
        mangocode_plugins::set_global_registry(plugin_registry.clone());
    }

    let mut existing_names: std::collections::HashSet<String> = config
        .mcp_servers
        .iter()
        .map(|server| server.name.clone())
        .collect();
    for mcp_server in plugin_registry.all_mcp_servers() {
        if existing_names.insert(mcp_server.name.clone()) {
            config.mcp_servers.push(mcp_server);
        }
    }

    config
}

async fn connect_acp_mcp_manager(config: &Config) -> Option<Arc<mangocode_mcp::McpManager>> {
    if config.mcp_servers.is_empty() {
        return None;
    }

    let manager = Arc::new(mangocode_mcp::McpManager::connect_all(&config.mcp_servers).await);
    manager.clone().spawn_notification_poll_loop();
    Some(manager)
}

fn build_acp_tools(
    mcp_manager: Option<Arc<mangocode_mcp::McpManager>>,
    config: &Config,
) -> Vec<Box<dyn mangocode_tools::Tool>> {
    let mut tools: Vec<Box<dyn mangocode_tools::Tool>> = mangocode_tools::all_tools();
    #[cfg(feature = "tool-agent")]
    tools.push(Box::new(mangocode_query::AgentTool));
    if let Some(manager) = mcp_manager {
        mangocode_tools::extend_with_mcp_tools(&mut tools, manager);
    }

    mangocode_tools::filter_tools_by_name_config(
        tools,
        &config.allowed_tools,
        &config.disallowed_tools,
    )
}

// ---------------------------------------------------------------------------
// Permission bridge
// ---------------------------------------------------------------------------

struct AcpPermissionHandler {
    session_id: String,
    mode: PermissionMode,
    out_tx: OutboundTx,
    state: Arc<AcpServerState>,
}

impl PermissionHandler for AcpPermissionHandler {
    fn check_permission(&self, request: &PermissionRequest) -> PermissionDecision {
        match self.mode {
            PermissionMode::BypassPermissions | PermissionMode::AcceptEdits => {
                PermissionDecision::Allow
            }
            PermissionMode::Plan => {
                if request.is_read_only {
                    PermissionDecision::Allow
                } else {
                    PermissionDecision::Deny
                }
            }
            PermissionMode::Default => {
                if request.is_read_only {
                    PermissionDecision::Allow
                } else {
                    PermissionDecision::Ask {
                        reason: request.description.clone(),
                    }
                }
            }
        }
    }

    fn request_permission(&self, request: &PermissionRequest) -> PermissionDecision {
        match self.check_permission(request) {
            PermissionDecision::Ask { .. } => {
                request_acp_permission(&self.session_id, request, &self.out_tx, &self.state)
            }
            other => other,
        }
    }
}

fn request_acp_permission(
    session_id: &str,
    request: &PermissionRequest,
    out_tx: &OutboundTx,
    state: &Arc<AcpServerState>,
) -> PermissionDecision {
    let request_id = format!("permission-{}", uuid::Uuid::new_v4());
    let (tx, rx) = std_mpsc::channel();
    state.pending_permissions.lock().unwrap().insert(
        request_id.clone(),
        PendingPermission {
            session_id: session_id.to_string(),
            tx,
        },
    );

    let sent = out_tx.send(json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/request_permission",
        "params": {
            "sessionId": session_id,
            "toolCall": {
                "toolCallId": request_id,
                "tool_call_id": request_id,
                "name": request.tool_name,
                "description": request.description,
                "details": request.details,
                "isReadOnly": request.is_read_only,
            },
            "options": [
                { "optionId": "allow-once", "name": "Allow once", "kind": "allow_once" },
                { "optionId": "allow-always", "name": "Always allow", "kind": "allow_always" },
                { "optionId": "reject-once", "name": "Reject", "kind": "reject_once" }
            ]
        }
    }));

    if sent.is_err() {
        state
            .pending_permissions
            .lock()
            .unwrap()
            .remove(&request_id);
        return PermissionDecision::Deny;
    }

    rx.recv_timeout(Duration::from_secs(60 * 60))
        .unwrap_or(PermissionDecision::Deny)
}

fn handle_client_response(resp: JsonRpcResponseMsg, state: &Arc<AcpServerState>) {
    let Some(id) = resp
        .id
        .as_ref()
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
    else {
        return;
    };
    if let Some(pending) = state.pending_permissions.lock().unwrap().remove(&id) {
        let decision = if resp.error.is_some() {
            PermissionDecision::Deny
        } else {
            permission_decision_from_result(resp.result)
        };
        let _ = pending.tx.send(decision);
    }
}

fn permission_decision_from_result(result: Option<Value>) -> PermissionDecision {
    let Some(result) = result else {
        return PermissionDecision::Deny;
    };
    let selected = result
        .get("selectedOptionId")
        .or_else(|| result.get("optionId"))
        .or_else(|| result.get("outcome").and_then(|o| o.get("optionId")))
        .and_then(Value::as_str)
        .unwrap_or("deny");
    match selected {
        "allow" | "allow-once" | "allow_once" => PermissionDecision::Allow,
        "allow_always" | "always_allow" | "allow-always" | "always-allow" => {
            PermissionDecision::AllowPermanently
        }
        _ => PermissionDecision::Deny,
    }
}

fn cancel_pending_permissions(state: &Arc<AcpServerState>, session_id: &str) {
    let mut pending = state.pending_permissions.lock().unwrap();
    let ids: Vec<String> = pending
        .iter()
        .filter_map(|(id, p)| {
            if p.session_id == session_id {
                Some(id.clone())
            } else {
                None
            }
        })
        .collect();
    for id in ids {
        if let Some(p) = pending.remove(&id) {
            let _ = p.tx.send(PermissionDecision::Deny);
        }
    }
}

// ---------------------------------------------------------------------------
// Notifications and event mapping
// ---------------------------------------------------------------------------

fn send_notification(out_tx: &OutboundTx, method: &str, params: Value) {
    let _ = out_tx.send(json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }));
}

fn send_session_update(out_tx: &OutboundTx, session_id: &str, update: Value) {
    send_notification(
        out_tx,
        "session/update",
        json!({
            "sessionId": session_id,
            "session_id": session_id,
            "update": update,
        }),
    );
}

fn acp_update_from_harness_event(event: &mangocode_core::harness::HarnessEvent) -> Option<Value> {
    match event.event_type.as_str() {
        "message.delta" => Some(json!({
            "sessionUpdate": "message_chunk",
            "role": "assistant",
            "content": {
                "type": "text",
                "text": event.payload.get("text").and_then(Value::as_str).unwrap_or_default()
            },
            "turnId": event.turn_id,
        })),
        "message.thinking_delta" => Some(json!({
            "sessionUpdate": "thinking_chunk",
            "content": {
                "type": "text",
                "text": event.payload.get("thinking").and_then(Value::as_str).unwrap_or_default()
            },
            "turnId": event.turn_id,
        })),
        "tool.started" => Some(json!({
            "sessionUpdate": "tool_call",
            "status": "started",
            "toolCallId": event.tool_call_id,
            "name": event.payload.get("tool_name").cloned().unwrap_or(Value::Null),
            "input": event.payload.get("input").cloned().unwrap_or(Value::Null),
            "turnId": event.turn_id,
        })),
        "tool.completed" => Some(json!({
            "sessionUpdate": "tool_call",
            "status": if event.payload.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
                "failed"
            } else {
                "completed"
            },
            "toolCallId": event.tool_call_id,
            "name": event.payload.get("tool_name").cloned().unwrap_or(Value::Null),
            "result": event.payload.get("result").cloned().unwrap_or(Value::Null),
            "turnId": event.turn_id,
        })),
        "turn.started" => Some(json!({
            "sessionUpdate": "status",
            "status": "started",
            "detail": event.payload,
            "turnId": event.turn_id,
        })),
        "turn.completed" => Some(json!({
            "sessionUpdate": "status",
            "status": "completed",
            "detail": event.payload,
            "turnId": event.turn_id,
        })),
        "turn.cancelled" => Some(json!({
            "sessionUpdate": "status",
            "status": "cancelled",
            "detail": event.payload,
            "turnId": event.turn_id,
        })),
        "turn.failed" => Some(json!({
            "sessionUpdate": "status",
            "status": "failed",
            "detail": event.payload,
            "turnId": event.turn_id,
        })),
        "file.changed" => Some(json!({
            "sessionUpdate": "diff",
            "path": event.payload.get("path").cloned().unwrap_or(Value::Null),
            "toolCallId": event.tool_call_id,
            "tool_call_id": event.tool_call_id,
            "beforeHash": event.payload.get("before_hash").cloned().unwrap_or(Value::Null),
            "afterHash": event.payload.get("after_hash").cloned().unwrap_or(Value::Null),
            "beforeText": event.payload.get("before_text").cloned().unwrap_or(Value::Null),
            "afterText": event.payload.get("after_text").cloned().unwrap_or(Value::Null),
            "binary": event.payload.get("binary").cloned().unwrap_or(Value::Bool(false)),
            "turnId": event.turn_id,
        })),
        "permission.decision" => Some(json!({
            "sessionUpdate": "permission",
            "status": "decided",
            "toolName": event.payload.get("tool_name").cloned().unwrap_or(Value::Null),
            "description": event.payload.get("description").cloned().unwrap_or(Value::Null),
            "decision": event.payload.get("decision").cloned().unwrap_or(Value::Null),
            "reason": event.payload.get("reason").cloned().unwrap_or(Value::Null),
            "isReadOnly": event.payload.get("is_read_only").cloned().unwrap_or(Value::Bool(false)),
            "turnId": event.turn_id,
        })),
        "permission.requested" => Some(json!({
            "sessionUpdate": "permission",
            "status": "requested",
            "toolName": event.payload.get("tool_name").cloned().unwrap_or(Value::Null),
            "description": event.payload.get("description").cloned().unwrap_or(Value::Null),
            "details": event.payload.get("details").cloned().unwrap_or(Value::Null),
            "isReadOnly": event.payload.get("is_read_only").cloned().unwrap_or(Value::Bool(false)),
            "turnId": event.turn_id,
        })),
        "status" => Some(json!({
            "sessionUpdate": "status",
            "message": event.payload.get("message").cloned().unwrap_or(Value::Null),
            "turnId": event.turn_id,
        })),
        "error" => Some(json!({
            "sessionUpdate": "error",
            "message": event.payload.get("message").cloned().unwrap_or(Value::Null),
            "turnId": event.turn_id,
        })),
        "checkpoint.before" | "checkpoint.after" => Some(json!({
            "sessionUpdate": "checkpoint",
            "kind": event.event_type.strip_prefix("checkpoint.").unwrap_or("checkpoint"),
            "checkpointId": event.checkpoint_id,
            "turnId": event.turn_id,
            "backend": event.payload.get("backend").cloned().unwrap_or(Value::Null),
        })),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn params_session_id(params: &Value) -> Option<String> {
    params
        .get("sessionId")
        .or_else(|| params.get("session_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn parse_permission_mode(mode: &str) -> Option<PermissionMode> {
    match mode {
        "default" | "ask" => Some(PermissionMode::Default),
        "accept_edits" | "accept-edits" | "code" => Some(PermissionMode::AcceptEdits),
        "bypass" | "bypass_permissions" | "bypass-permissions" => {
            Some(PermissionMode::BypassPermissions)
        }
        "plan" | "architect" => Some(PermissionMode::Plan),
        _ => None,
    }
}

fn permission_mode_id(mode: &PermissionMode) -> &'static str {
    match mode {
        PermissionMode::Default => "ask",
        PermissionMode::AcceptEdits => "code",
        PermissionMode::BypassPermissions => "bypass",
        PermissionMode::Plan => "architect",
    }
}

fn modes_json(current: &PermissionMode) -> Value {
    json!({
        "currentModeId": permission_mode_id(current),
        "availableModes": [
            {
                "id": "ask",
                "name": "Ask",
                "description": "Ask before making changes"
            },
            {
                "id": "architect",
                "name": "Plan",
                "description": "Read and plan without editing files"
            },
            {
                "id": "code",
                "name": "Code",
                "description": "Allow file edits while still blocking dangerous commands"
            },
            {
                "id": "bypass",
                "name": "Bypass",
                "description": "Allow tools without prompting"
            }
        ]
    })
}

fn extract_prompt_text(params: &Value) -> Option<String> {
    if let Some(text) = params
        .get("prompt")
        .and_then(Value::as_str)
        .or_else(|| params.get("message").and_then(Value::as_str))
        .or_else(|| params.get("content").and_then(Value::as_str))
    {
        return Some(text.to_string());
    }

    for candidate in [
        params.get("prompt"),
        params.get("content"),
        params.get("message"),
        params.get("text"),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(text) = collect_text_content(candidate) {
            return Some(text);
        }
    }

    None
}

fn collect_text_content(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let text = items
                .iter()
                .filter_map(collect_text_content)
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            if matches!(
                map.get("type").and_then(Value::as_str),
                Some("text" | "Text")
            ) {
                if let Some(text) = map.get("content").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
            map.get("content")
                .or_else(|| map.get("blocks"))
                .or_else(|| map.get("value"))
                .and_then(collect_text_content)
        }
        _ => None,
    }
}

fn role_name(message: &Message) -> &'static str {
    match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn message_text(message: &Message) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

async fn tool_list_json(
    params: Option<Value>,
    state: Arc<AcpServerState>,
) -> anyhow::Result<Value> {
    let params = params.unwrap_or(Value::Null);
    let (cwd, model_override, mode_override) = if let Some(session_id) = params_session_id(&params)
    {
        let session_arc = {
            state
                .sessions
                .lock()
                .unwrap()
                .get(&session_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Unknown session: {}", session_id))?
        };
        let session = session_arc.lock().await;
        (
            session.cwd.clone(),
            session.model_override.clone(),
            session.permission_mode.clone(),
        )
    } else {
        (
            params
                .get("cwd")
                .or_else(|| params.get("workingDirectory"))
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
            params
                .get("model")
                .or_else(|| params.get("modelId"))
                .or_else(|| params.get("model_id"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            params
                .get("modeId")
                .or_else(|| params.get("mode_id"))
                .or_else(|| params.get("mode"))
                .and_then(Value::as_str)
                .and_then(parse_permission_mode),
        )
    };

    let config = acp_effective_config(&cwd, model_override, mode_override, false).await;
    let mcp_manager = connect_acp_mcp_manager(&config).await;
    let tools = build_acp_tools(mcp_manager, &config);
    Ok(tool_list_value_from_tools(&tools))
}

fn tool_list_value_from_tools(tools: &[Box<dyn mangocode_tools::Tool>]) -> Value {
    let mut specs = mangocode_tools::build_registry_plan(tools).specs;
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    json!({ "tools": specs })
}

fn model_list_json() -> Value {
    let registry = mangocode_api::ModelRegistry::new();
    let mut entries = registry.list_all();
    entries.sort_by(|a, b| {
        (*a.info.provider_id)
            .cmp(&*b.info.provider_id)
            .then_with(|| (*a.info.id).cmp(&*b.info.id))
    });
    let models: Vec<_> = entries
        .iter()
        .map(|e| {
            json!({
                "id": format!("{}/{}", e.info.provider_id, e.info.id),
                "name": e.info.name,
                "context_window": e.info.context_window,
                "provider": e.info.provider_id.to_string(),
            })
        })
        .collect();
    json!({ "models": models })
}

async fn list_sessions_json() -> Value {
    let sessions = mangocode_core::history::list_sessions().await;
    if !sessions.is_empty() {
        let arr: Vec<_> = sessions
            .iter()
            .map(|s| {
                json!({
                    "id": s.id,
                    "sessionId": s.id,
                    "session_id": s.id,
                    "title": s.title,
                    "model": s.model,
                    "created_at": s.created_at.to_rfc3339(),
                    "updated_at": s.updated_at.to_rfc3339(),
                    "message_count": s.messages.len(),
                    "working_dir": s.working_dir,
                })
            })
            .collect();
        return Value::Array(arr);
    }
    try_list_sessions()
}

fn try_list_sessions() -> Value {
    let db_path = Settings::config_dir().join("sessions.db");
    match mangocode_core::SqliteSessionStore::open(&db_path) {
        Ok(store) => match store.list_sessions() {
            Ok(sessions) => {
                let arr: Vec<_> = sessions
                    .iter()
                    .map(|s| {
                        json!({
                            "id": s.id,
                            "sessionId": s.id,
                            "session_id": s.id,
                            "title": s.title,
                            "model": s.model,
                            "created_at": s.created_at,
                            "updated_at": s.updated_at,
                            "message_count": s.message_count,
                        })
                    })
                    .collect();
                Value::Array(arr)
            }
            Err(e) => {
                warn!(error = %e, "ACP: failed to list sessions");
                Value::Array(vec![])
            }
        },
        Err(e) => {
            warn!(error = %e, "ACP: failed to open SQLite store");
            Value::Array(vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_legacy_message_prompt() {
        let params = json!({ "session_id": "s", "message": "hello" });
        assert_eq!(extract_prompt_text(&params).as_deref(), Some("hello"));
    }

    #[test]
    fn extracts_official_content_block_prompt() {
        let params = json!({
            "sessionId": "s",
            "prompt": {
                "content": [
                    { "type": "text", "text": "hello" },
                    { "type": "text", "content": "world" }
                ]
            }
        });
        assert_eq!(
            extract_prompt_text(&params).as_deref(),
            Some("hello\nworld")
        );
    }

    #[test]
    fn initialize_uses_integer_protocol_version() {
        let result = initialize_result(Some(json!({ "protocolVersion": 1 })));
        assert_eq!(result["protocolVersion"].as_u64(), Some(1));
        assert_eq!(
            result["agentCapabilities"]["loadSession"].as_bool(),
            Some(true)
        );
        assert!(result["authMethods"].as_array().is_some());
    }

    #[test]
    fn maps_permission_result() {
        let decision = permission_decision_from_result(Some(json!({
            "selectedOptionId": "allow-once"
        })));
        assert_eq!(decision, PermissionDecision::Allow);

        let decision = permission_decision_from_result(Some(json!({
            "selectedOptionId": "allow-always"
        })));
        assert_eq!(decision, PermissionDecision::AllowPermanently);
    }

    #[cfg(all(feature = "tool-bash", feature = "tool-tool-search"))]
    #[test]
    fn acp_tool_filter_respects_config_aliases() {
        let mut tools = mangocode_tools::all_tools();
        #[cfg(feature = "tool-agent")]
        tools.push(Box::new(mangocode_query::AgentTool));

        let filtered = mangocode_tools::filter_tools_by_name_config(
            tools,
            &["shell_command".to_string(), "ToolSearch".to_string()],
            &["container.exec".to_string()],
        );

        assert!(filtered.iter().any(|tool| tool.name() == "ToolSearch"));
        assert!(!filtered.iter().any(|tool| tool.name() == "Bash"));
        assert!(!filtered.iter().any(|tool| tool.name() == "Read"));
    }

    #[test]
    fn acp_tool_list_matches_runtime_registry() {
        #[allow(unused_mut)]
        let mut runtime_tools = mangocode_tools::all_tools();
        #[cfg(feature = "tool-agent")]
        runtime_tools.push(Box::new(mangocode_query::AgentTool));
        let expected = mangocode_tools::build_registry_plan(&runtime_tools)
            .specs
            .into_iter()
            .map(|spec| spec.name)
            .collect::<std::collections::BTreeSet<_>>();

        let advertised = tool_list_value_from_tools(&runtime_tools);
        let tools = advertised["tools"]
            .as_array()
            .expect("tools/list should return a tools array");
        let actual = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str().map(str::to_string))
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(actual, expected);

        let default_tool_set_enabled = cfg!(any(
            feature = "default-tools",
            feature = "default-tools-no-web-research",
            feature = "full-tools"
        ));
        let tool_search_enabled =
            default_tool_set_enabled || cfg!(feature = "tool-tool-search");
        let view_image_enabled =
            default_tool_set_enabled || cfg!(feature = "tool-view-image");
        let get_goal_enabled = default_tool_set_enabled || cfg!(feature = "tool-get-goal");
        let create_goal_enabled =
            default_tool_set_enabled || cfg!(feature = "tool-create-goal");
        let update_goal_enabled =
            default_tool_set_enabled || cfg!(feature = "tool-update-goal");
        let agent_enabled = default_tool_set_enabled || cfg!(feature = "tool-agent");

        for (name, should_exist) in [
            ("ToolSearch", tool_search_enabled),
            ("ViewImage", view_image_enabled),
            ("get_goal", get_goal_enabled),
            ("create_goal", create_goal_enabled),
            ("update_goal", update_goal_enabled),
            ("Agent", agent_enabled),
        ] {
            assert_eq!(
                actual.contains(name),
                should_exist,
                "ACP advertised tool presence should match feature gate for {name}"
            );
        }

        #[allow(unused_variables)]
        let aliases_for = |name: &str| -> Vec<String> {
            tools
                .iter()
                .find(|tool| tool["name"].as_str() == Some(name))
                .and_then(|tool| tool["aliases"].as_array())
                .expect("tool should include aliases")
                .iter()
                .filter_map(|alias| alias.as_str().map(str::to_string))
                .collect()
        };
        if default_tool_set_enabled || cfg!(feature = "tool-bash") {
            assert!(aliases_for("Bash").contains(&"shell_command".to_string()));
        }
        if view_image_enabled {
            assert!(aliases_for("ViewImage").contains(&"view_image".to_string()));
        }
        if tool_search_enabled {
            assert!(aliases_for("ToolSearch").contains(&"tool_search".to_string()));
        }
        if agent_enabled {
            assert!(aliases_for("Agent").contains(&"spawn_agent".to_string()));
        }
    }

    #[cfg(any(
        feature = "tool-tool-search",
        feature = "default-tools",
        feature = "default-tools-no-web-research",
        feature = "full-tools"
    ))]
    #[tokio::test]
    async fn acp_tool_list_uses_session_runtime_config() {
        let dir = std::env::temp_dir().join(format!("mangocode-acp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join(".mangocode")).unwrap();
        std::fs::write(
            dir.join(".mangocode").join("settings.json"),
            r#"{ "config": { "allowed_tools": ["ToolSearch"], "disallowed_tools": ["Bash"] } }"#,
        )
        .unwrap();

        let state = Arc::new(AcpServerState::default());
        create_session(
            Some(json!({
                "sessionId": "tool-list-session",
                "cwd": dir.display().to_string()
            })),
            state.clone(),
        )
        .await
        .unwrap();

        let advertised = tool_list_json(
            Some(json!({ "sessionId": "tool-list-session" })),
            state.clone(),
        )
        .await
        .unwrap();
        let tools = advertised["tools"]
            .as_array()
            .expect("tools/list should return a tools array");
        let names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<std::collections::BTreeSet<_>>();

        assert!(names.contains("ToolSearch"));
        assert!(!names.contains("Bash"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn maps_turn_and_file_change_events_to_session_updates() {
        let event = mangocode_core::harness::HarnessEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            session_id: "s".to_string(),
            turn_id: Some("t".to_string()),
            tool_call_id: None,
            checkpoint_id: None,
            event_type: "turn.completed".to_string(),
            timestamp: chrono::Utc::now(),
            payload: json!({ "stop_reason": "end_turn" }),
        };
        let update = acp_update_from_harness_event(&event).unwrap();
        assert_eq!(update["sessionUpdate"].as_str(), Some("status"));
        assert_eq!(update["status"].as_str(), Some("completed"));

        let event = mangocode_core::harness::HarnessEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            session_id: "s".to_string(),
            turn_id: Some("t".to_string()),
            tool_call_id: Some("tool-1".to_string()),
            checkpoint_id: None,
            event_type: "file.changed".to_string(),
            timestamp: chrono::Utc::now(),
            payload: json!({
                "path": "src/main.rs",
                "before_text": "old",
                "after_text": "new",
                "binary": false
            }),
        };
        let update = acp_update_from_harness_event(&event).unwrap();
        assert_eq!(update["sessionUpdate"].as_str(), Some("diff"));
        assert_eq!(update["toolCallId"].as_str(), Some("tool-1"));
        assert_eq!(update["path"].as_str(), Some("src/main.rs"));

        let event = mangocode_core::harness::HarnessEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            session_id: "s".to_string(),
            turn_id: Some("t".to_string()),
            tool_call_id: None,
            checkpoint_id: None,
            event_type: "permission.decision".to_string(),
            timestamp: chrono::Utc::now(),
            payload: json!({
                "tool_name": "Write",
                "description": "Write file",
                "decision": "allow"
            }),
        };
        let update = acp_update_from_harness_event(&event).unwrap();
        assert_eq!(update["sessionUpdate"].as_str(), Some("permission"));
        assert_eq!(update["status"].as_str(), Some("decided"));
        assert_eq!(update["decision"].as_str(), Some("allow"));
    }

    #[tokio::test]
    async fn set_model_and_mode_update_session_runtime_inputs() {
        let state = Arc::new(AcpServerState::default());
        create_session(Some(json!({ "sessionId": "default-model" })), state.clone())
            .await
            .unwrap();
        {
            let session = state
                .sessions
                .lock()
                .unwrap()
                .get("default-model")
                .cloned()
                .unwrap();
            assert!(session.lock().await.model_override.is_none());
        }

        let created = create_session(
            Some(json!({
                "sessionId": "s",
                "modelId": "anthropic/claude-sonnet-4-5",
                "modeId": "ask"
            })),
            state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(created["sessionId"].as_str(), Some("s"));

        set_session_model(
            Some(json!({
                "sessionId": "s",
                "modelId": "anthropic/claude-opus-4-5"
            })),
            state.clone(),
        )
        .await
        .unwrap();
        set_session_mode(
            Some(json!({
                "sessionId": "s",
                "modeId": "architect"
            })),
            state.clone(),
        )
        .await
        .unwrap();

        let session = state.sessions.lock().unwrap().get("s").cloned().unwrap();
        let session = session.lock().await;
        assert_eq!(session.session.model, "anthropic/claude-opus-4-5");
        assert_eq!(
            session.model_override.as_deref(),
            Some("anthropic/claude-opus-4-5")
        );
        assert_eq!(session.permission_mode, Some(PermissionMode::Plan));
    }
}
