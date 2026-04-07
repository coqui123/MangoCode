// AgentTool: spawn a sub-agent to handle a complex sub-task.
//
// Lives in cc-query (not cc-tools) to avoid a circular dependency:
//   cc-tools would need cc-query, but cc-query already needs cc-tools.
//
// The AgentTool creates a nested query loop with its own context, enabling
// the model to delegate complex work to specialized sub-agents. Each sub-agent:
//   - Runs its own agentic loop
//   - Has access to all tools (except AgentTool itself, preventing infinite recursion)
//   - Returns its final output as the tool result
//
// New capabilities (TS parity):
//   - `isolation: "worktree"` - run the agent in a dedicated git worktree so
//     file edits don't conflict with the parent checkout or sibling agents.
//   - `run_in_background: true` - fire-and-forget; returns agent_id immediately.
//     Use poll_background_agent() to check completion status.

use async_trait::async_trait;
use dashmap::DashMap;
use mangocode_api::client::ClientConfig;
use mangocode_api::AnthropicClient;
use mangocode_core::types::Message;
use mangocode_tools::{PermissionLevel, Tool, ToolContext, ToolResult};
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::{run_query_loop, QueryConfig, QueryOutcome};

// ---------------------------------------------------------------------------
// Background agent registry
// ---------------------------------------------------------------------------

/// Registry of in-flight background agents.
/// Maps agent_id -> oneshot receiver that resolves to the agent's final output.
static BACKGROUND_AGENTS: Lazy<DashMap<String, tokio::sync::oneshot::Receiver<String>>> =
    Lazy::new(DashMap::new);

/// Poll a background agent's result.
///
/// Returns `None` if still running, `Some(result_text)` when done (or errored).
/// After returning `Some`, the entry is removed from the registry.
pub fn poll_background_agent(agent_id: &str) -> Option<String> {
    if let Some(mut entry) = BACKGROUND_AGENTS.get_mut(agent_id) {
        match entry.try_recv() {
            Ok(result) => {
                drop(entry);
                BACKGROUND_AGENTS.remove(agent_id);
                Some(result)
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
            Err(_) => {
                // Sender dropped - treat as agent error/cancellation.
                drop(entry);
                BACKGROUND_AGENTS.remove(agent_id);
                Some("[Agent error or cancelled]".to_string())
            }
        }
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Worktree isolation helpers
// ---------------------------------------------------------------------------

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

async fn create_worktree(git_root: &Path, agent_id: &str) -> Option<PathBuf> {
    let worktree_dir = std::env::temp_dir().join(format!("claude-agent-{}", agent_id));
    let output = tokio::process::Command::new("git")
        .args([
            "worktree",
            "add",
            "--detach",
            worktree_dir.to_str().unwrap_or_default(),
            "HEAD",
        ])
        .current_dir(git_root)
        .output()
        .await
        .ok()?;
    if output.status.success() {
        Some(worktree_dir)
    } else {
        warn!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        None
    }
}

async fn remove_worktree(git_root: &Path, worktree_dir: &Path) {
    let _ = tokio::process::Command::new("git")
        .args([
            "worktree",
            "remove",
            "--force",
            worktree_dir.to_str().unwrap_or_default(),
        ])
        .current_dir(git_root)
        .output()
        .await;
}

fn build_provider_and_model_registries(
    ctx: &ToolContext,
    credential: String,
    use_bearer_auth: bool,
) -> (
    std::sync::Arc<mangocode_api::ProviderRegistry>,
    std::sync::Arc<mangocode_api::ModelRegistry>,
) {
    let provider_registry = std::sync::Arc::new(
        mangocode_api::ProviderRegistry::from_environment_with_auth_store(ClientConfig {
            api_key: credential,
            api_base: ctx.config.resolve_api_base(),
            use_bearer_auth,
            ..Default::default()
        }),
    );

    let mut model_registry = mangocode_api::ModelRegistry::new();
    if let Some(cache_dir) = dirs::cache_dir() {
        let cache_path = cache_dir.join("mangocode").join("models_dev.json");
        model_registry.load_cache(&cache_path);
    }

    (provider_registry, std::sync::Arc::new(model_registry))
}

// ---------------------------------------------------------------------------
// AgentTool
// ---------------------------------------------------------------------------

pub struct AgentTool;

#[derive(Debug, Deserialize)]
struct AgentInput {
    /// Short description of the agent's task (used for logging).
    description: String,
    /// The complete task prompt to send as the first user message.
    prompt: String,
    /// Optional: which tools to make available (defaults to all minus AgentTool).
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Optional: system prompt override for the sub-agent.
    #[serde(default)]
    system_prompt: Option<String>,
    /// Optional: max turns for the sub-agent (default 10).
    #[serde(default)]
    max_turns: Option<u32>,
    /// Optional: model override for this sub-agent.
    #[serde(default)]
    model: Option<String>,
    /// Set to "worktree" to run the agent in an isolated git worktree.
    /// Omit (or set to null) for shared working directory.
    #[serde(default)]
    isolation: Option<String>,
    /// If true, start the agent in the background and return agent_id immediately.
    /// Default: false (wait for completion).
    #[serde(default)]
    run_in_background: bool,
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_AGENT
    }

    fn description(&self) -> &str {
        "Launch a new agent to handle complex, multi-step tasks autonomously. \
         The agent runs its own agentic loop with access to tools and returns \
         its final result. Use this to delegate sub-tasks, run parallel \
         workstreams, or handle tasks that require many tool calls."
    }

    fn permission_level(&self) -> PermissionLevel {
        // The agent inherits parent permissions; no extra level required.
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of the agent's task (3-5 words)"
                },
                "prompt": {
                    "type": "string",
                    "description": "The complete task for the agent to perform"
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tool names to make available. Defaults to all tools."
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt override for the sub-agent"
                },
                "max_turns": {
                    "type": "number",
                    "description": "Maximum number of turns for the sub-agent (default 10)"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model to use for this agent"
                },
                "isolation": {
                    "type": "string",
                    "enum": ["worktree"],
                    "description": "Set to \"worktree\" to run the agent in an isolated git worktree. \
                                    Prevents file-edit conflicts when multiple agents run in parallel."
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "If true, the agent starts immediately and this call returns an \
                                    agent_id without waiting for completion. Poll with poll_background_agent \
                                    to retrieve the result. Default: false."
                }
            },
            "required": ["description", "prompt"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: AgentInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        info!(description = %params.description, "Spawning sub-agent");

        let (credential, use_bearer_auth) = ctx
            .config
            .resolve_auth_async()
            .await
            .unwrap_or_else(|| (String::new(), false));

        // Dedicated Anthropic client for the sub-agent.
        let client = match AnthropicClient::new(ClientConfig {
            api_key: credential.clone(),
            api_base: ctx.config.resolve_api_base(),
            use_bearer_auth,
            ..Default::default()
        }) {
            Ok(c) => Arc::new(c),
            Err(e) => return ToolResult::error(format!("Failed to create client: {}", e)),
        };

        let (provider_registry, model_registry) =
            build_provider_and_model_registries(ctx, credential, use_bearer_auth);

        // Build the tool list for the sub-agent.
        // Always exclude AgentTool itself to prevent unbounded recursion.
        let all = mangocode_tools::all_tools();
        let agent_tools: Vec<Box<dyn Tool>> = if let Some(ref allowed) = params.tools {
            all.into_iter()
                .filter(|t| allowed.contains(&t.name().to_string()))
                .collect()
        } else {
            all.into_iter()
                .filter(|t| t.name() != mangocode_core::constants::TOOL_NAME_AGENT)
                .collect()
        };

        // Resolve model: explicit override > default.
        let model = params
            .model
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| ctx.config.effective_model().to_string());

        let system_prompt = params.system_prompt.unwrap_or_else(|| {
            let mut prompt = "You are a specialized AI agent helping with a specific sub-task. \
             Complete the task thoroughly and return your findings."
                .to_string();

            // Append plugin-contributed agent definitions so the sub-agent
            // is aware of any specialised agents declared by plugins.
            if let Some(registry) = mangocode_plugins::global_plugin_registry() {
                let mut agent_defs = String::new();
                for agent_dir in registry.all_agent_paths() {
                    if let Ok(entries) = std::fs::read_dir(&agent_dir) {
                        for entry in entries.flatten() {
                            let p = entry.path();
                            if p.extension().is_some_and(|e| e == "md") {
                                if let Ok(content) = std::fs::read_to_string(&p) {
                                    let name =
                                        p.file_stem().and_then(|s| s.to_str()).unwrap_or("agent");
                                    agent_defs.push_str(&format!(
                                        "\n\n## Agent: {}\n{}",
                                        name,
                                        content.trim()
                                    ));
                                }
                            }
                        }
                    }
                }
                if !agent_defs.is_empty() {
                    prompt.push_str("\n\nThe following specialized agents are available:");
                    prompt.push_str(&agent_defs);
                }
            }

            prompt
        });

        // -----------------------------------------------------------------------
        // Determine working directory - optionally isolate in a git worktree.
        // -----------------------------------------------------------------------
        let use_isolation = params.isolation.as_deref() == Some("worktree");
        let agent_id = uuid::Uuid::new_v4().to_string();

        let (working_dir_str, worktree_path, git_root): (String, Option<PathBuf>, Option<PathBuf>) =
            if use_isolation {
                let git_root = find_git_root(&ctx.working_dir);
                if let Some(ref root) = git_root {
                    if let Some(wt) = create_worktree(root, &agent_id).await {
                        let wd = wt.display().to_string();
                        (wd, Some(wt), git_root)
                    } else {
                        warn!(
                            agent_id = %agent_id,
                            "Worktree creation failed; running agent in shared working directory"
                        );
                        (ctx.working_dir.display().to_string(), None, None)
                    }
                } else {
                    warn!(
                        agent_id = %agent_id,
                        "No git root found; isolation=worktree ignored"
                    );
                    (ctx.working_dir.display().to_string(), None, None)
                }
            } else {
                (ctx.working_dir.display().to_string(), None, None)
            };

        let query_config = QueryConfig {
            model,
            max_tokens: mangocode_core::constants::DEFAULT_MAX_TOKENS,
            max_turns: params.max_turns.unwrap_or(10),
            system_prompt: Some(system_prompt),
            append_system_prompt: None,
            output_style: ctx.config.effective_output_style(),
            output_style_prompt: ctx.config.resolve_output_style_prompt(),
            working_directory: Some(working_dir_str),
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
            provider_registry: Some(provider_registry),
            agent_name: None,
            agent_definition: None,
            model_registry: Some(model_registry),
        };
        // -----------------------------------------------------------------------
        // Background mode: spawn and return agent_id immediately.
        // -----------------------------------------------------------------------
        if params.run_in_background {
            return spawn_background_agent(BackgroundAgentSpec {
                agent_id,
                description: params.description.clone(),
                prompt: params.prompt.clone(),
                agent_tools_bg: agent_tools,
                client_bg: client.clone(),
                ctx_bg: ctx.clone(),
                config_bg: query_config,
                git_root,
                worktree_path,
            });
        }

        // -----------------------------------------------------------------------
        // Synchronous mode: run the sub-agent loop and wait for completion.
        // -----------------------------------------------------------------------
        let mut messages = vec![Message::user(params.prompt)];
        let cancel = CancellationToken::new();

        let outcome = run_query_loop(
            client.as_ref(),
            &mut messages,
            &agent_tools,
            ctx,
            &query_config,
            ctx.cost_tracker.clone(),
            None, // no event forwarding for sub-agents
            cancel,
            None, // no pending message queue for sub-agents
        )
        .await;

        // Cleanup worktree if one was created.
        if let (Some(root), Some(wt)) = (git_root, worktree_path) {
            remove_worktree(&root, &wt).await;
        }

        match outcome {
            QueryOutcome::EndTurn { message, usage } => {
                let text = message.get_all_text();
                debug!(
                    description = %params.description,
                    output_tokens = usage.output_tokens,
                    "Sub-agent completed"
                );
                ToolResult::success(text)
            }
            QueryOutcome::MaxTokens {
                partial_message, ..
            } => {
                let text = partial_message.get_all_text();
                ToolResult::success(format!("{}\n\n[Note: Agent hit max_tokens limit]", text))
            }
            QueryOutcome::Cancelled => ToolResult::error("Sub-agent was cancelled".to_string()),
            QueryOutcome::Error(e) => ToolResult::error(format!("Sub-agent error: {}", e)),
            QueryOutcome::BudgetExceeded {
                cost_usd,
                limit_usd,
            } => ToolResult::error(format!(
                "Sub-agent stopped: budget ${:.4} exceeded (limit ${:.4})",
                cost_usd, limit_usd
            )),
        }
    }
}

struct BackgroundAgentSpec {
    agent_id: String,
    description: String,
    prompt: String,
    agent_tools_bg: Vec<Box<dyn Tool>>,
    client_bg: Arc<AnthropicClient>,
    ctx_bg: ToolContext,
    config_bg: QueryConfig,
    git_root: Option<PathBuf>,
    worktree_path: Option<PathBuf>,
}

fn spawn_background_agent(spec: BackgroundAgentSpec) -> ToolResult {
    let BackgroundAgentSpec {
        agent_id,
        description,
        prompt,
        agent_tools_bg,
        client_bg,
        ctx_bg,
        config_bg,
        git_root,
        worktree_path,
    } = spec;

    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    BACKGROUND_AGENTS.insert(agent_id.clone(), rx);

    let description_for_msg = description.clone();
    let description_for_log = description;
    let agent_id_for_log = agent_id.clone();

    tokio::spawn(async move {
        let cancel = CancellationToken::new();
        let mut messages = vec![Message::user(prompt)];
        let outcome = run_query_loop(
            client_bg.as_ref(),
            &mut messages,
            &agent_tools_bg,
            &ctx_bg,
            &config_bg,
            ctx_bg.cost_tracker.clone(),
            None,
            cancel,
            None,
        )
        .await;

        if let (Some(root), Some(wt)) = (git_root, worktree_path) {
            remove_worktree(&root, &wt).await;
        }

        let result_text = format_outcome(outcome);
        debug!(
            agent_id = %agent_id_for_log,
            description = %description_for_log,
            "Background agent completed"
        );
        let _ = tx.send(result_text);
    });

    ToolResult::success(
        serde_json::json!({
            "agent_id": agent_id,
            "status": "running",
            "message": format!(
                "Agent '{}' started in background. Use poll_background_agent with agent_id '{}' to check status.",
                description_for_msg, agent_id
            )
        })
        .to_string(),
    )
}

/// Execute AgentTool using the active query runtime so subagents can inherit
/// the current provider setup (Vertex/OpenAI/etc.) instead of requiring a
/// separate Anthropic-only client.
pub async fn execute_with_runtime(
    input: Value,
    ctx: &ToolContext,
    runtime_client: &mangocode_api::AnthropicClient,
    parent_query_config: &QueryConfig,
    parent_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::QueryEvent>>,
    parent_tool_use_id: Option<String>,
) -> ToolResult {
    let params: AgentInput = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
    };

    info!(description = %params.description, "Spawning sub-agent");

    // Build the tool list for the sub-agent.
    // Always exclude AgentTool itself to prevent unbounded recursion.
    let all = mangocode_tools::all_tools();
    let agent_tools: Vec<Box<dyn Tool>> = if let Some(ref allowed) = params.tools {
        all.into_iter()
            .filter(|t| allowed.contains(&t.name().to_string()))
            .collect()
    } else {
        all.into_iter()
            .filter(|t| t.name() != mangocode_core::constants::TOOL_NAME_AGENT)
            .collect()
    };

    // Resolve model: explicit override > parent query model > default.
    let model = params
        .model
        .clone()
        .filter(|m| !m.is_empty())
        .or_else(|| {
            (!parent_query_config.model.is_empty()).then(|| parent_query_config.model.clone())
        })
        .unwrap_or_else(|| mangocode_core::constants::DEFAULT_MODEL.to_string());

    let system_prompt = params.system_prompt.clone().unwrap_or_else(|| {
        let mut prompt = "You are a specialized AI agent helping with a specific sub-task. \
             Complete the task thoroughly and return your findings."
            .to_string();

        if let Some(registry) = mangocode_plugins::global_plugin_registry() {
            let mut agent_defs = String::new();
            for agent_dir in registry.all_agent_paths() {
                if let Ok(entries) = std::fs::read_dir(&agent_dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.extension().is_some_and(|e| e == "md") {
                            if let Ok(content) = std::fs::read_to_string(&p) {
                                let name =
                                    p.file_stem().and_then(|s| s.to_str()).unwrap_or("agent");
                                agent_defs.push_str(&format!(
                                    "\n\n## Agent: {}\n{}",
                                    name,
                                    content.trim()
                                ));
                            }
                        }
                    }
                }
            }
            if !agent_defs.is_empty() {
                prompt.push_str("\n\nThe following specialized agents are available:");
                prompt.push_str(&agent_defs);
            }
        }

        prompt
    });

    // Determine working directory - optionally isolate in a git worktree.
    let use_isolation = params.isolation.as_deref() == Some("worktree");
    let agent_id = uuid::Uuid::new_v4().to_string();

    let (working_dir_str, worktree_path, git_root): (String, Option<PathBuf>, Option<PathBuf>) =
        if use_isolation {
            let git_root = find_git_root(&ctx.working_dir);
            if let Some(ref root) = git_root {
                if let Some(wt) = create_worktree(root, &agent_id).await {
                    let wd = wt.display().to_string();
                    (wd, Some(wt), git_root)
                } else {
                    warn!(
                        agent_id = %agent_id,
                        "Worktree creation failed; running agent in shared working directory"
                    );
                    (ctx.working_dir.display().to_string(), None, None)
                }
            } else {
                warn!(
                    agent_id = %agent_id,
                    "No git root found; isolation=worktree ignored"
                );
                (ctx.working_dir.display().to_string(), None, None)
            }
        } else {
            (ctx.working_dir.display().to_string(), None, None)
        };

    let mut query_config = parent_query_config.clone();
    query_config.model = model;
    query_config.max_tokens = mangocode_core::constants::DEFAULT_MAX_TOKENS;
    query_config.max_turns = params.max_turns.unwrap_or(10);
    query_config.system_prompt = Some(system_prompt);
    query_config.append_system_prompt = None;
    query_config.output_style = ctx.config.effective_output_style();
    query_config.output_style_prompt = ctx.config.resolve_output_style_prompt();
    query_config.working_directory = Some(working_dir_str);
    query_config.command_queue = None;
    query_config.skill_index = None;
    query_config.agent_name = None;
    query_config.agent_definition = None;

    // Background mode: launch a detached sub-agent while preserving the active
    // provider/model runtime context from the parent query config.
    if params.run_in_background {
        let (credential, use_bearer_auth) = ctx
            .config
            .resolve_auth_async()
            .await
            .unwrap_or_else(|| (String::new(), false));

        let client_bg = match AnthropicClient::new(ClientConfig {
            api_key: credential,
            api_base: ctx.config.resolve_api_base(),
            use_bearer_auth,
            ..Default::default()
        }) {
            Ok(c) => Arc::new(c),
            Err(e) => return ToolResult::error(format!("Failed to create client: {}", e)),
        };

        return spawn_background_agent(BackgroundAgentSpec {
            agent_id,
            description: params.description.clone(),
            prompt: params.prompt.clone(),
            agent_tools_bg: agent_tools,
            client_bg,
            ctx_bg: ctx.clone(),
            config_bg: query_config,
            git_root,
            worktree_path,
        });
    }

    // Foreground mode: optionally forward child events and attach parent tool id.
    let (event_tx, forward_task) = if let Some(parent_tx) = parent_event_tx {
        let (sub_tx, mut sub_rx) = tokio::sync::mpsc::unbounded_channel::<crate::QueryEvent>();
        let parent_id = parent_tool_use_id.clone();
        let fwd = tokio::spawn(async move {
            while let Some(evt) = sub_rx.recv().await {
                let mapped = if let Some(ref pid) = parent_id {
                    attach_parent_to_event(evt, pid)
                } else {
                    evt
                };
                let _ = parent_tx.send(mapped);
            }
        });
        (Some(sub_tx), Some(fwd))
    } else {
        (None, None)
    };

    let mut messages = vec![Message::user(params.prompt)];
    let cancel = CancellationToken::new();

    let outcome = run_query_loop(
        runtime_client,
        &mut messages,
        &agent_tools,
        ctx,
        &query_config,
        ctx.cost_tracker.clone(),
        event_tx.clone(),
        cancel,
        None,
    )
    .await;

    drop(event_tx);
    if let Some(task) = forward_task {
        let _ = task.await;
    }

    // Cleanup worktree if one was created.
    if let (Some(root), Some(wt)) = (git_root, worktree_path) {
        remove_worktree(&root, &wt).await;
    }

    match outcome {
        QueryOutcome::EndTurn { message, usage } => {
            let text = message.get_all_text();
            debug!(
                description = %params.description,
                output_tokens = usage.output_tokens,
                "Sub-agent completed"
            );
            ToolResult::success(text)
        }
        QueryOutcome::MaxTokens {
            partial_message, ..
        } => {
            let text = partial_message.get_all_text();
            ToolResult::success(format!("{}\n\n[Note: Agent hit max_tokens limit]", text))
        }
        QueryOutcome::Cancelled => ToolResult::error("Sub-agent was cancelled".to_string()),
        QueryOutcome::Error(e) => ToolResult::error(format!("Sub-agent error: {}", e)),
        QueryOutcome::BudgetExceeded {
            cost_usd,
            limit_usd,
        } => ToolResult::error(format!(
            "Sub-agent stopped: budget ${:.4} exceeded (limit ${:.4})",
            cost_usd, limit_usd
        )),
    }
}

fn attach_parent_to_event(
    event: crate::QueryEvent,
    default_parent_tool_use_id: &str,
) -> crate::QueryEvent {
    match event {
        crate::QueryEvent::Stream(event) => crate::QueryEvent::StreamWithParent {
            event,
            parent_tool_use_id: default_parent_tool_use_id.to_string(),
        },
        crate::QueryEvent::StreamWithParent {
            event,
            parent_tool_use_id,
        } => crate::QueryEvent::StreamWithParent {
            event,
            parent_tool_use_id,
        },
        crate::QueryEvent::ToolStart {
            tool_name,
            tool_id,
            input_json,
            parent_tool_use_id,
        } => crate::QueryEvent::ToolStart {
            tool_name,
            tool_id,
            input_json,
            parent_tool_use_id: Some(
                parent_tool_use_id.unwrap_or_else(|| default_parent_tool_use_id.to_string()),
            ),
        },
        crate::QueryEvent::ToolEnd {
            tool_name,
            tool_id,
            result,
            is_error,
            parent_tool_use_id,
        } => crate::QueryEvent::ToolEnd {
            tool_name,
            tool_id,
            result,
            is_error,
            parent_tool_use_id: Some(
                parent_tool_use_id.unwrap_or_else(|| default_parent_tool_use_id.to_string()),
            ),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Helper: convert a QueryOutcome into a result string for background agents
// ---------------------------------------------------------------------------

fn format_outcome(outcome: QueryOutcome) -> String {
    match outcome {
        QueryOutcome::EndTurn { message, .. } => message.get_all_text(),
        QueryOutcome::MaxTokens {
            partial_message, ..
        } => format!(
            "{}\n\n[Note: Agent hit max_tokens limit]",
            partial_message.get_all_text()
        ),
        QueryOutcome::Cancelled => "[Agent was cancelled]".to_string(),
        QueryOutcome::Error(e) => format!("[Agent error: {}]", e),
        QueryOutcome::BudgetExceeded {
            cost_usd,
            limit_usd,
        } => format!(
            "[Agent stopped: budget ${:.4} exceeded (limit ${:.4})]",
            cost_usd, limit_usd
        ),
    }
}

// ---------------------------------------------------------------------------
// Team swarm runner injection
// ---------------------------------------------------------------------------
//
// Called once at process startup (e.g. from main.rs) to inject a real agent
// runner into cc-tools so that TeamCreateTool can spawn sub-agents via
// run_query_loop without creating a circular crate dependency.

/// Register the cc-query-backed agent runner with cc-tools.
///
/// After this call, `TeamCreateTool` will actually invoke `run_query_loop` for
/// each agent instead of returning stub output.
///
/// # Panics
/// Panics if the runner was already registered.
pub fn init_team_swarm_runner() {
    let runner: mangocode_tools::AgentRunFn = Arc::new(
        |description: String,
         prompt: String,
         tools: Option<Vec<String>>,
         system: Option<String>,
         max_turns: Option<u32>,
         ctx: Arc<mangocode_tools::ToolContext>| {
            // We must return a Pin<Box<dyn Future<...> + Send>>.
            Box::pin(async move {
                let (credential, use_bearer_auth) = ctx
                    .config
                    .resolve_auth_async()
                    .await
                    .unwrap_or_else(|| (String::new(), false));

                let client =
                    match mangocode_api::AnthropicClient::new(mangocode_api::client::ClientConfig {
                        api_key: credential.clone(),
                        api_base: ctx.config.resolve_api_base(),
                        use_bearer_auth,
                        ..Default::default()
                    }) {
                        Ok(c) => Arc::new(c),
                        Err(e) => {
                            return format!(
                                "[Agent '{}' failed to create client: {}]",
                                description, e
                            )
                        }
                    };

                let (provider_registry, model_registry) =
                    build_provider_and_model_registries(&ctx, credential, use_bearer_auth);

                // Build the tool list, filtering to the allowlist if provided.
                let all = mangocode_tools::all_tools();
                let agent_tools: Vec<Box<dyn mangocode_tools::Tool>> =
                    if let Some(ref allowed) = tools {
                        all.into_iter()
                            .filter(|t| allowed.contains(&t.name().to_string()))
                            .collect()
                    } else {
                        all.into_iter()
                            .filter(|t| t.name() != mangocode_core::constants::TOOL_NAME_AGENT)
                            .collect()
                    };

                let model = ctx.config.effective_model().to_string();

                let system_prompt = system.unwrap_or_else(|| {
                    "You are a specialized AI agent helping with a specific sub-task. \
                     Complete the task thoroughly and return your findings."
                        .to_string()
                });

                let query_config = crate::QueryConfig {
                    model,
                    max_tokens: mangocode_core::constants::DEFAULT_MAX_TOKENS,
                    max_turns: max_turns.unwrap_or(10),
                    system_prompt: Some(system_prompt),
                    working_directory: Some(ctx.working_dir.display().to_string()),
                    output_style: ctx.config.effective_output_style(),
                    output_style_prompt: ctx.config.resolve_output_style_prompt(),
                    provider_registry: Some(provider_registry),
                    model_registry: Some(model_registry),
                    ..Default::default()
                };

                let cancel = tokio_util::sync::CancellationToken::new();
                let mut messages = vec![mangocode_core::types::Message::user(prompt)];
                let outcome = crate::run_query_loop(
                    client.as_ref(),
                    &mut messages,
                    &agent_tools,
                    &ctx,
                    &query_config,
                    ctx.cost_tracker.clone(),
                    None,
                    cancel,
                    None,
                )
                .await;

                format_outcome(outcome)
            }) as Pin<Box<dyn std::future::Future<Output = String> + Send>>
        },
    );

    mangocode_tools::register_agent_runner(runner);
}

#[cfg(test)]
mod tests {
    use super::attach_parent_to_event;

    #[test]
    fn attaches_parent_to_plain_stream_events() {
        let evt =
            crate::QueryEvent::Stream(mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                index: 0,
                delta: mangocode_api::streaming::ContentDelta::TextDelta {
                    text: "hello".to_string(),
                },
            });

        let out = attach_parent_to_event(evt, "parent-1");
        match out {
            crate::QueryEvent::StreamWithParent {
                parent_tool_use_id, ..
            } => assert_eq!(parent_tool_use_id, "parent-1"),
            _ => panic!("expected StreamWithParent"),
        }
    }

    #[test]
    fn preserves_existing_parent_on_tool_events() {
        let evt = crate::QueryEvent::ToolStart {
            tool_name: "Bash".to_string(),
            tool_id: "tool-1".to_string(),
            input_json: "{}".to_string(),
            parent_tool_use_id: Some("existing-parent".to_string()),
        };

        let out = attach_parent_to_event(evt, "fallback-parent");
        match out {
            crate::QueryEvent::ToolStart {
                parent_tool_use_id, ..
            } => assert_eq!(parent_tool_use_id.as_deref(), Some("existing-parent")),
            _ => panic!("expected ToolStart"),
        }
    }

    #[test]
    fn applies_fallback_parent_on_tool_end() {
        let evt = crate::QueryEvent::ToolEnd {
            tool_name: "Read".to_string(),
            tool_id: "tool-2".to_string(),
            result: "ok".to_string(),
            is_error: false,
            parent_tool_use_id: None,
        };

        let out = attach_parent_to_event(evt, "fallback-parent");
        match out {
            crate::QueryEvent::ToolEnd {
                parent_tool_use_id, ..
            } => assert_eq!(parent_tool_use_id.as_deref(), Some("fallback-parent")),
            _ => panic!("expected ToolEnd"),
        }
    }
}
