//! Integration tests for the full query loop using MockProvider.

use async_trait::async_trait;
use mangocode_api::client::ClientConfig;
use mangocode_api::providers::mock::ToolCall;
use mangocode_api::providers::MockProvider;
use mangocode_api::ProviderRegistry;
use mangocode_core::config::{Config as CoreConfig, PermissionMode};
use mangocode_core::cost::CostTracker;
use mangocode_core::permissions::AutoPermissionHandler;
use mangocode_core::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent};
use mangocode_query::{run_query_loop, QueryConfig, QueryOutcome};
use mangocode_tools::{PermissionLevel, Tool, ToolContext, ToolResult};
use serde_json::json;
use std::sync::atomic::AtomicUsize;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo_tool"
    }

    fn description(&self) -> &str {
        "Echo input"
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        })
    }

    async fn execute(&self, input: serde_json::Value, _ctx: &ToolContext) -> ToolResult {
        let value = input
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        ToolResult::success(format!("echo:{}", value))
    }
}

fn make_client() -> mangocode_api::AnthropicClient {
    mangocode_api::AnthropicClient::new(ClientConfig {
        api_key: "test-key".to_string(),
        api_base: "https://example.invalid".to_string(),
        use_bearer_auth: false,
        max_retries: 0,
        request_timeout: std::time::Duration::from_secs(1),
        initial_retry_delay: std::time::Duration::from_millis(1),
        max_retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    })
    .expect("client")
}

fn make_registry(mock: MockProvider) -> std::sync::Arc<ProviderRegistry> {
    let mut registry = ProviderRegistry::new();
    registry.register(std::sync::Arc::new(mock));
    std::sync::Arc::new(registry)
}

fn test_config(registry: std::sync::Arc<ProviderRegistry>) -> QueryConfig {
    QueryConfig {
        model: "mock/mock-model".to_string(),
        max_tokens: 200_000,
        max_turns: 10,
        system_prompt: Some("You are a test assistant.".to_string()),
        provider_registry: Some(registry),
        ..Default::default()
    }
}

fn test_tool_ctx(provider: &str) -> ToolContext {
    let cfg = CoreConfig {
        provider: Some(provider.to_string()),
        ..Default::default()
    };
    ToolContext {
        working_dir: std::env::temp_dir(),
        permission_mode: PermissionMode::BypassPermissions,
        permission_handler: std::sync::Arc::new(AutoPermissionHandler {
            mode: PermissionMode::BypassPermissions,
        }),
        cost_tracker: CostTracker::new(),
        session_metrics: None,
        session_id: "test-session".to_string(),
        file_history: std::sync::Arc::new(parking_lot::Mutex::new(
            mangocode_core::file_history::FileHistory::new(),
        )),
        current_turn: std::sync::Arc::new(AtomicUsize::new(0)),
        non_interactive: true,
        mcp_manager: None,
        config: cfg,
    }
}

fn has_assistant_text(messages: &[Message], needle: &str) -> bool {
    messages.iter().any(|m| {
        m.role == Role::Assistant
            && match &m.content {
                MessageContent::Text(t) => t.contains(needle),
                MessageContent::Blocks(blocks) => blocks.iter().any(|b| {
                    matches!(b, ContentBlock::Text { text } if text.contains(needle))
                }),
            }
    })
}

#[tokio::test]
async fn simple_text_response() {
    let provider = MockProvider::with_responses(vec!["Hello from mock!"]);
    let config = test_config(make_registry(provider));
    let ctx = test_tool_ctx("mock");
    let mut messages = vec![Message::user("Hi there")];

    let outcome = run_query_loop(
        &make_client(),
        &mut messages,
        &[],
        &ctx,
        &config,
        CostTracker::new(),
        None,
        tokio_util::sync::CancellationToken::new(),
        None,
    )
    .await;

    match outcome {
        QueryOutcome::EndTurn { message, .. } => {
            assert!(message.get_all_text().contains("Hello from mock!"));
        }
        other => panic!("Expected EndTurn, got {:?}", other),
    }
}

#[tokio::test]
async fn single_tool_call_chain() {
    let provider = MockProvider::with_responses(vec!["Done reading the file."]).with_tool_sequence(
        vec![
            vec![ToolCall::new("tool-1", "echo_tool", json!({ "value": "one" }))],
            vec![],
        ],
    );

    let config = test_config(make_registry(provider.clone()));
    let ctx = test_tool_ctx("mock");
    let mut messages = vec![Message::user("Run echo tool")];
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];

    let outcome = run_query_loop(
        &make_client(),
        &mut messages,
        &tools,
        &ctx,
        &config,
        CostTracker::new(),
        None,
        tokio_util::sync::CancellationToken::new(),
        None,
    )
    .await;

    match outcome {
        QueryOutcome::EndTurn { message, .. } => {
            assert!(message.get_all_text().contains("Done reading"));
        }
        other => panic!("Expected EndTurn, got {:?}", other),
    }

    assert!(
        messages.len() >= 3,
        "Expected at least 3 messages (user + assistant tool_use + tool_result + final)"
    );
    assert_eq!(provider.request_count(), 2);
}

#[tokio::test]
async fn context_overflow_triggers_compaction() {
    let big = "x".repeat(15_000);
    let mut messages = vec![
        Message::user("Start"),
        Message::assistant("Thinking"),
        Message::user_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: ToolResultContent::Text(big.clone()),
                is_error: Some(false),
            },
            ContentBlock::ToolResult {
                tool_use_id: "t2".to_string(),
                content: ToolResultContent::Text(big.clone()),
                is_error: Some(false),
            },
        ]),
        Message::user("Final question"),
    ];

    let provider = MockProvider::with_responses(vec!["Compacted response"]);
    let mut config = test_config(make_registry(provider));
    config.tool_result_budget = 200;
    let ctx = test_tool_ctx("mock");

    let _ = run_query_loop(
        &make_client(),
        &mut messages,
        &[],
        &ctx,
        &config,
        CostTracker::new(),
        None,
        tokio_util::sync::CancellationToken::new(),
        None,
    )
    .await;

    let saw_truncation = messages.iter().any(|m| match &m.content {
        MessageContent::Blocks(blocks) => blocks.iter().any(|b| {
            matches!(
                b,
                ContentBlock::ToolResult {
                    content: ToolResultContent::Text(text),
                    ..
                } if text.contains("[tool result truncated to save context]")
            )
        }),
        _ => false,
    });

    assert!(saw_truncation, "Expected tool-result budget compaction");
}

#[tokio::test]
async fn multi_step_tool_chain() {
    let provider = MockProvider::with_responses(vec!["Found and read the file."]).with_tool_sequence(
        vec![
            vec![ToolCall::new("t1", "echo_tool", json!({ "value": "a" }))],
            vec![ToolCall::new("t2", "echo_tool", json!({ "value": "b" }))],
            vec![],
        ],
    );

    let config = test_config(make_registry(provider.clone()));
    let ctx = test_tool_ctx("mock");
    let mut messages = vec![Message::user("Find and read txt files")];
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];

    let outcome = run_query_loop(
        &make_client(),
        &mut messages,
        &tools,
        &ctx,
        &config,
        CostTracker::new(),
        None,
        tokio_util::sync::CancellationToken::new(),
        None,
    )
    .await;

    match outcome {
        QueryOutcome::EndTurn { .. } => {}
        other => panic!("Expected EndTurn, got {:?}", other),
    }

    assert!(has_assistant_text(&messages, "Found and read the file."));
    assert_eq!(provider.request_count(), 3);
}
