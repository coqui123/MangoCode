//! Context Collapse Service
//!
//! Automatically reduces conversation size to fit within model context windows.
//! Uses multiple compaction strategies, including local and model-assisted
//! summarization.

use crate::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent};
use crate::{FeatureFlags, FLAG_LLM_COMPACTION};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Strategy for collapsing a conversation when it exceeds token limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollapseStrategy {
    /// Drop oldest non-system messages first
    DropOldest,
    /// Replace verbose tool results with one-line snippets
    SnipCompact,
    /// Incrementally trim very long messages
    MicroCompact,
    /// Summarize the dropped middle by calling an API client
    LlmSummarize,
    /// Chain strategies from least to most aggressive
    ReactiveCompact,
}

/// Minimal API interface used for LLM-based context summarization.
pub trait ApiClient {
    fn summarize<'a>(
        &'a self,
        model: &'a str,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>>;
}

/// Collapse state persisted to disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseState {
    pub session_id: String,
    pub messages_dropped: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub strategy_used: String,
    pub collapsed_at: String,
}

/// Natural-language average chars/token.
const TEXT_CHARS_PER_TOKEN: f64 = 4.0;
/// Code-heavy average chars/token.
const CODE_CHARS_PER_TOKEN: f64 = 2.8;

/// Estimate token count from text using a code-aware heuristic.
fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 1;
    }

    let divisor = if looks_like_code(text) {
        CODE_CHARS_PER_TOKEN
    } else {
        TEXT_CHARS_PER_TOKEN
    };

    ((text.len() as f64 / divisor).ceil() as u64).max(1)
}

/// Estimate total tokens in a message list.
pub fn estimate_message_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens_single).sum()
}

/// Collapse a message list to fit within max_tokens.
/// Returns the collapsed message list and collapse state (if collapsing occurred).
pub fn collapse_context(
    messages: Vec<Message>,
    max_tokens: u64,
    strategy: CollapseStrategy,
) -> (Vec<Message>, Option<CollapseState>) {
    let original_messages = messages.clone();
    let initial_tokens = estimate_message_tokens(&messages);

    // Already under limit
    if initial_tokens <= max_tokens {
        return (messages, None);
    }

    let (collapsed, changed_count, strategy_used) = match strategy {
        CollapseStrategy::DropOldest => {
            let (m, changed) = drop_oldest_messages(messages, max_tokens);
            (m, changed, CollapseStrategy::DropOldest)
        }
        CollapseStrategy::SnipCompact => {
            let (m, changed) = snip_compact(messages, max_tokens);
            (m, changed, CollapseStrategy::SnipCompact)
        }
        CollapseStrategy::MicroCompact => {
            let (m, changed) = micro_compact(messages, max_tokens);
            (m, changed, CollapseStrategy::MicroCompact)
        }
        CollapseStrategy::LlmSummarize => {
            // The synchronous API does not have an ApiClient handle, so use
            // the local reactive fallback chain instead of forcing DropOldest.
            let (m, changed, used) = reactive_compact_local(messages, max_tokens);
            (m, changed, used)
        }
        CollapseStrategy::ReactiveCompact => {
            let (m, changed, used) = reactive_compact_local(messages, max_tokens);
            (m, changed, used)
        }
    };

    let final_tokens = estimate_message_tokens(&collapsed);

    if final_tokens >= initial_tokens {
        return (original_messages, None);
    }

    let state = CollapseState {
        session_id: "unknown".to_string(),
        messages_dropped: changed_count,
        tokens_before: initial_tokens,
        tokens_after: final_tokens,
        strategy_used: format!("{:?}", strategy_used),
        collapsed_at: chrono::Utc::now().to_rfc3339(),
    };

    (collapsed, Some(state))
}

/// LLM summarize compaction:
/// keeps first 3 and last 5 messages and replaces middle with a summary message.
pub async fn llm_summarize_compact(
    messages: Vec<Message>,
    max_tokens: u64,
    api_client: &dyn ApiClient,
    model: &str,
) -> (Vec<Message>, Option<CollapseState>) {
    if !FeatureFlags::is_enabled(FLAG_LLM_COMPACTION) {
        return (messages, None);
    }

    let initial_tokens = estimate_message_tokens(&messages);
    if initial_tokens <= max_tokens || messages.len() <= 8 {
        return (messages, None);
    }

    let head_keep = 3usize.min(messages.len());
    let tail_keep = 5usize.min(messages.len().saturating_sub(head_keep));
    let middle_start = head_keep;
    let middle_end = messages.len().saturating_sub(tail_keep);

    if middle_start >= middle_end {
        return (messages, None);
    }

    let head = messages[..middle_start].to_vec();
    let middle = messages[middle_start..middle_end].to_vec();
    let tail = messages[middle_end..].to_vec();

    let mut transcript = String::new();
    for msg in &middle {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        transcript.push_str(role);
        transcript.push_str(": ");
        transcript.push_str(&message_to_text(msg));
        transcript.push_str("\n\n");
    }

    let prompt = format!(
        "Summarize this conversation segment in under 500 words. Preserve: all file paths mentioned, all decisions made, all errors encountered, and all pending tasks.\n\n{}",
        transcript
    );

    let summary = match api_client.summarize(model, prompt).await {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return (messages, None),
    };

    let mut collapsed = Vec::with_capacity(head.len() + 1 + tail.len());
    collapsed.extend(head);
    collapsed.push(Message::assistant(format!("[System summary]\n{}", summary.trim())));
    collapsed.extend(tail);

    let final_tokens = estimate_message_tokens(&collapsed);
    if final_tokens >= initial_tokens {
        return (messages, None);
    }

    let state = CollapseState {
        session_id: "unknown".to_string(),
        messages_dropped: middle.len(),
        tokens_before: initial_tokens,
        tokens_after: final_tokens,
        strategy_used: format!("{:?}", CollapseStrategy::LlmSummarize),
        collapsed_at: chrono::Utc::now().to_rfc3339(),
    };

    (collapsed, Some(state))
}

/// Reactive compaction chains strategies in increasing aggressiveness:
/// SnipCompact -> MicroCompact -> DropOldest -> LlmSummarize.
pub async fn reactive_compact(
    messages: Vec<Message>,
    max_tokens: u64,
    api_client: &dyn ApiClient,
    model: &str,
) -> (Vec<Message>, Option<CollapseState>) {
    let initial_tokens = estimate_message_tokens(&messages);
    if initial_tokens <= max_tokens {
        return (messages, None);
    }

    let (snipped, snip_count) = snip_compact(messages, max_tokens);
    if estimate_message_tokens(&snipped) <= max_tokens {
        return (
            snipped.clone(),
            Some(build_state(
                initial_tokens,
                estimate_message_tokens(&snipped),
                snip_count,
                CollapseStrategy::SnipCompact,
            )),
        );
    }

    let (microed, micro_count) = micro_compact(snipped, max_tokens);
    if estimate_message_tokens(&microed) <= max_tokens {
        return (
            microed.clone(),
            Some(build_state(
                initial_tokens,
                estimate_message_tokens(&microed),
                micro_count,
                CollapseStrategy::MicroCompact,
            )),
        );
    }

    let (dropped, dropped_count) = drop_oldest_messages(microed, max_tokens);
    if estimate_message_tokens(&dropped) <= max_tokens {
        return (
            dropped.clone(),
            Some(build_state(
                initial_tokens,
                estimate_message_tokens(&dropped),
                dropped_count,
                CollapseStrategy::DropOldest,
            )),
        );
    }

    let (llm, state) = llm_summarize_compact(dropped, max_tokens, api_client, model).await;
    if let Some(mut state) = state {
        state.tokens_before = initial_tokens;
        state.strategy_used = format!("{:?}", CollapseStrategy::LlmSummarize);
        return (llm, Some(state));
    }

    (llm, None)
}

/// Drop oldest non-system messages until under token limit.
fn drop_oldest_messages(mut messages: Vec<Message>, max_tokens: u64) -> (Vec<Message>, usize) {
    let mut dropped = 0;

    // Find first non-system user/assistant message (skip system roles)
    let first_user_idx = messages
        .iter()
        .position(|m| m.role != Role::Assistant) // Keep assistant responses, drop user turns
        .unwrap_or(0);

    // Drop messages starting from first_user_idx
    while estimate_message_tokens(&messages) > max_tokens && messages.len() > first_user_idx + 1 {
        messages.remove(first_user_idx);
        dropped += 1;
    }

    (messages, dropped)
}

/// Replace tool results with one-line snippets where possible.
fn snip_compact(mut messages: Vec<Message>, _max_tokens: u64) -> (Vec<Message>, usize) {
    let mut tool_names: HashMap<String, String> = HashMap::new();
    for msg in &messages {
        if let MessageContent::Blocks(blocks) = &msg.content {
            for block in blocks {
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    tool_names.insert(id.clone(), name.clone());
                }
            }
        }
    }

    let mut changed = 0usize;
    for msg in &mut messages {
        if msg.role != Role::Assistant {
            continue;
        }

        if let MessageContent::Blocks(blocks) = &mut msg.content {
            for block in blocks {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                {
                    let tool_name = tool_names
                        .get(tool_use_id)
                        .map(String::as_str)
                        .unwrap_or("unknown_tool");
                    let preview = first_n_chars(&tool_result_to_text(content), 80);
                    let has_more = tool_result_to_text(content).chars().count() > 80;
                    let suffix = if has_more { "..." } else { "" };
                    *content = ToolResultContent::Text(format!(
                        "[Tool result from {}: {}{}]",
                        tool_name,
                        preview.replace('\n', " "),
                        suffix
                    ));
                    changed += 1;
                }
            }
        }
    }

    (messages, changed)
}

/// Incrementally trim the single longest message until under limit.
fn micro_compact(mut messages: Vec<Message>, max_tokens: u64) -> (Vec<Message>, usize) {
    let mut changed = 0usize;

    loop {
        if estimate_message_tokens(&messages) <= max_tokens {
            break;
        }

        let mut longest_idx = None;
        let mut longest_tokens = 0u64;

        for (idx, msg) in messages.iter().enumerate() {
            let t = estimate_message_tokens_single(msg);
            if t > longest_tokens {
                longest_tokens = t;
                longest_idx = Some(idx);
            }
        }

        let Some(idx) = longest_idx else {
            break;
        };

        if longest_tokens <= 500 {
            break;
        }

        if !truncate_message_middle(&mut messages[idx]) {
            break;
        }

        changed += 1;
    }

    (messages, changed)
}

fn reactive_compact_local(
    messages: Vec<Message>,
    max_tokens: u64,
) -> (Vec<Message>, usize, CollapseStrategy) {
    let (snipped, snip_count) = snip_compact(messages, max_tokens);
    if estimate_message_tokens(&snipped) <= max_tokens {
        return (snipped, snip_count, CollapseStrategy::SnipCompact);
    }

    let (microed, micro_count) = micro_compact(snipped, max_tokens);
    if estimate_message_tokens(&microed) <= max_tokens {
        return (microed, micro_count, CollapseStrategy::MicroCompact);
    }

    let (dropped, dropped_count) = drop_oldest_messages(microed, max_tokens);
    (dropped, dropped_count, CollapseStrategy::DropOldest)
}

fn build_state(
    tokens_before: u64,
    tokens_after: u64,
    changed_count: usize,
    strategy: CollapseStrategy,
) -> CollapseState {
    CollapseState {
        session_id: "unknown".to_string(),
        messages_dropped: changed_count,
        tokens_before,
        tokens_after,
        strategy_used: format!("{:?}", strategy),
        collapsed_at: chrono::Utc::now().to_rfc3339(),
    }
}

fn looks_like_code(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let code_markers = [
        "fn ",
        "def ",
        "class ",
        "import ",
        "let ",
        "const ",
        "=>",
        "{",
        "}",
        "::",
        "pub ",
        "use ",
        "#include",
        "function ",
    ];

    if code_markers.iter().any(|m| lowered.contains(m)) {
        return true;
    }

    let punct = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | ';' | '(' | ')' | '[' | ']'))
        .count();
    let lines = text.lines().count().max(1);
    punct / lines >= 2
}

fn estimate_message_tokens_single(msg: &Message) -> u64 {
    let content_tokens: u64 = match &msg.content {
        MessageContent::Text(t) => estimate_tokens(t),
        MessageContent::Blocks(blocks) => blocks.iter().map(estimate_block_tokens).sum(),
    };
    content_tokens + 2
}

fn estimate_block_tokens(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } => estimate_tokens(text),
        ContentBlock::ToolUse { name, input, .. } => {
            estimate_tokens(&format!("{} {}", name, input))
        }
        ContentBlock::ToolResult { content, .. } => estimate_tokens(&tool_result_to_text(content)),
        ContentBlock::Thinking { thinking, .. } => estimate_tokens(thinking),
        ContentBlock::RedactedThinking { data } => estimate_tokens(data),
        ContentBlock::UserLocalCommandOutput { command, output } => {
            estimate_tokens(&format!("{} {}", command, output))
        }
        ContentBlock::UserCommand { name, args } => estimate_tokens(&format!("{} {}", name, args)),
        ContentBlock::UserMemoryInput { key, value } => {
            estimate_tokens(&format!("{} {}", key, value))
        }
        ContentBlock::SystemAPIError { message, .. } => estimate_tokens(message),
        ContentBlock::CollapsedReadSearch {
            tool_name,
            paths,
            n_hidden,
        } => estimate_tokens(&format!("{} {} {}", tool_name, paths.join(" "), n_hidden)),
        ContentBlock::TaskAssignment {
            id,
            subject,
            description,
        } => estimate_tokens(&format!("{} {} {}", id, subject, description)),
        ContentBlock::Image { .. } | ContentBlock::Document { .. } => 250,
    }
}

fn truncate_message_middle(msg: &mut Message) -> bool {
    match &mut msg.content {
        MessageContent::Text(text) => {
            let truncated = truncate_text_middle(text);
            if truncated == *text {
                false
            } else {
                *text = truncated;
                true
            }
        }
        MessageContent::Blocks(_) => {
            let as_text = message_to_text(msg);
            let truncated = truncate_text_middle(&as_text);
            if truncated == as_text {
                false
            } else {
                msg.content = MessageContent::Text(truncated);
                true
            }
        }
    }
}

fn truncate_text_middle(text: &str) -> String {
    let total_chars = text.chars().count();
    if total_chars < 60 {
        return text.to_string();
    }

    let keep = ((total_chars as f64) * 0.2).ceil() as usize;
    let keep_each = keep.max(1);
    if keep_each * 2 >= total_chars {
        return text.to_string();
    }

    let head = first_n_chars(text, keep_each);
    let tail = last_n_chars(text, keep_each);

    let total_tokens = estimate_tokens(text);
    let kept_tokens = estimate_tokens(&head) + estimate_tokens(&tail);
    let snipped = total_tokens.saturating_sub(kept_tokens).max(1);

    format!("{}\n[...{} tokens snipped...]\n{}", head, snipped, tail)
}

fn first_n_chars(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn last_n_chars(text: &str, n: usize) -> String {
    let len = text.chars().count();
    text.chars().skip(len.saturating_sub(n)).collect()
}

fn tool_result_to_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(t) => t.clone(),
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .map(block_to_text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn message_to_text(msg: &Message) -> String {
    match &msg.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .map(block_to_text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn block_to_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {} {}", name, input),
        ContentBlock::ToolResult { content, .. } => tool_result_to_text(content),
        ContentBlock::Thinking { thinking, .. } => thinking.clone(),
        ContentBlock::RedactedThinking { data } => data.clone(),
        ContentBlock::UserLocalCommandOutput { command, output } => {
            format!("{}\n{}", command, output)
        }
        ContentBlock::UserCommand { name, args } => format!("{} {}", name, args),
        ContentBlock::UserMemoryInput { key, value } => format!("{}: {}", key, value),
        ContentBlock::SystemAPIError { message, .. } => message.clone(),
        ContentBlock::CollapsedReadSearch {
            tool_name,
            paths,
            n_hidden,
        } => format!("{} {} (+{})", tool_name, paths.join(","), n_hidden),
        ContentBlock::TaskAssignment {
            id,
            subject,
            description,
        } => format!("{} {} {}", id, subject, description),
        ContentBlock::Image { .. } => "[image]".to_string(),
        ContentBlock::Document { .. } => "[document]".to_string(),
    }
}

/// Persist collapse state to ~/.mangocode/context_collapse_state.json
pub fn save_collapse_state(_session_id: &str, state: &CollapseState) -> anyhow::Result<()> {
    let path = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?
        .join(".mangocode")
        .join("context_collapse_state.json");

    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Collapse state path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let json = serde_json::to_string(state)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Load collapse state from ~/.mangocode/context_collapse_state.json
pub fn load_collapse_state(_session_id: &str) -> Option<CollapseState> {
    let path = dirs::home_dir()?
        .join(".mangocode")
        .join("context_collapse_state.json");

    if !path.exists() {
        return None;
    }

    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockApiClient {
        summary: String,
    }

    impl ApiClient for MockApiClient {
        fn summarize<'a>(
            &'a self,
            _model: &'a str,
            _prompt: String,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
            Box::pin(async move { Ok(self.summary.clone()) })
        }
    }

    fn long_code_message() -> Message {
        let mut code = String::new();
        for i in 0..300 {
            code.push_str(&format!("fn thing_{}() {{ let x = {}; }}\n", i, i));
        }
        Message::assistant(code)
    }

    #[test]
    fn token_estimation_is_code_aware() {
        let code = "fn main() { let x = 1; println!(\"{}\", x); }\n".repeat(40);
        let prose = "This is a plain sentence with mostly normal language content.\n".repeat(40);
        assert!(estimate_tokens(&code) > estimate_tokens(&prose));
    }

    #[test]
    fn snip_compact_replaces_tool_result_blocks() {
        let messages = vec![Message::assistant_blocks(vec![
            ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "src/main.rs"}),
            },
            ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: ToolResultContent::Text("A".repeat(300)),
                is_error: None,
            },
        ])];

        let (out, changed) = snip_compact(messages, 200);
        assert_eq!(changed, 1);

        let MessageContent::Blocks(blocks) = &out[0].content else {
            panic!("expected block content");
        };
        let ContentBlock::ToolResult { content, .. } = &blocks[1] else {
            panic!("expected tool result block");
        };
        let ToolResultContent::Text(text) = content else {
            panic!("expected text tool result");
        };
        assert!(text.starts_with("[Tool result from read_file:"));
    }

    #[test]
    fn micro_compact_trims_longest_message() {
        let messages = vec![Message::user("small"), long_code_message(), Message::assistant("ok")];
        let before = estimate_message_tokens(&messages);
        let (out, changed) = micro_compact(messages, before.saturating_sub(300));
        assert!(changed > 0);
        assert!(estimate_message_tokens(&out) < before);

        let text = out[1].get_all_text();
        assert!(text.contains("tokens snipped"));
    }

    #[tokio::test]
    async fn llm_summarize_compact_keeps_head_and_tail() {
        let client = MockApiClient {
            summary: "summary payload".to_string(),
        };

        let messages = (0..14)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(format!("user {}", i))
                } else {
                    Message::assistant(format!("assistant {}", i))
                }
            })
            .collect::<Vec<_>>();

        let (out, state) = llm_summarize_compact(messages.clone(), 10, &client, "test-model").await;
        assert!(state.is_some());
        assert_eq!(out.len(), 9);
        assert_eq!(out[0].get_all_text(), messages[0].get_all_text());
        assert_eq!(out[1].get_all_text(), messages[1].get_all_text());
        assert_eq!(out[2].get_all_text(), messages[2].get_all_text());
        assert!(out[3].get_all_text().contains("summary payload"));
    }

    #[tokio::test]
    async fn reactive_compact_chains_until_under_limit() {
        let client = MockApiClient {
            summary: "reactive summary".to_string(),
        };
        let messages = vec![
            Message::assistant_blocks(vec![
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "grep_search".to_string(),
                    input: serde_json::json!({"query": "foo"}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tool-1".to_string(),
                    content: ToolResultContent::Text("line\n".repeat(400)),
                    is_error: None,
                },
            ]),
            long_code_message(),
            Message::user("follow-up request"),
            Message::assistant("another reply"),
            Message::user("more context"),
            Message::assistant("tail 1"),
            Message::user("tail 2"),
            Message::assistant("tail 3"),
            Message::user("tail 4"),
            Message::assistant("tail 5"),
        ];

        let limit = estimate_message_tokens(&messages) / 3;
        let (out, state) = reactive_compact(messages, limit, &client, "test-model").await;
        assert!(state.is_some());
        assert!(estimate_message_tokens(&out) <= limit || out.len() < 10);
    }

    #[test]
    fn collapse_context_updates_strategy_used() {
        let messages = vec![long_code_message(), Message::assistant("tail")];
        let (out, state) = collapse_context(
            messages.clone(),
            estimate_message_tokens(&messages) / 2,
            CollapseStrategy::MicroCompact,
        );
        assert!(state.is_some());
        assert!(estimate_message_tokens(&out) < estimate_message_tokens(&messages));
        let state = state.expect("state expected");
        assert_eq!(state.strategy_used, "MicroCompact");
    }
}
