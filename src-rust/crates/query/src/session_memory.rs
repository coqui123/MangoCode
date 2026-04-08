// Session Memory Extraction for cc-query.
//
// Runs a background task after a session to extract key facts worth
// remembering and persist them to AGENTS.md.
//
// This mirrors TypeScript services/SessionMemory/sessionMemory.ts and
// services/extractMemories/extractMemories.ts.
//
// Strategy:
//   1. After sessions with 20+ messages (or on compact), call the API with a
//      structured extraction prompt.
//   2. Parse the response into typed ExtractedMemory entries.
//   3. Append entries under "## Auto-extracted memories" in AGENTS.md
//      (creating the file if it doesn't exist).
//   4. Track state so we don't re-extract from already-processed messages.

use mangocode_api::{
    AnthropicStreamEvent, ApiMessage, CreateMessageRequest, StreamAccumulator, StreamHandler,
    SystemPrompt,
};
use mangocode_core::types::{Message, Role};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Threshold constants (mirrors TypeScript sessionMemoryUtils.ts defaults)
// ---------------------------------------------------------------------------

/// Minimum messages before extraction is attempted.
const MIN_MESSAGES_TO_EXTRACT: usize = 20;

/// Minimum tool calls since last extraction before we run again.
const MIN_TOOL_CALLS_BETWEEN_EXTRACTIONS: usize = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Category of an extracted memory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryCategory {
    UserPreference,
    ProjectFact,
    CodePattern,
    Decision,
    Constraint,
}

impl MemoryCategory {
    /// Parse from the string tag produced by the model.
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "user_preference" | "userpreference" | "preference" => Self::UserPreference,
            "project_fact" | "projectfact" | "fact" => Self::ProjectFact,
            "code_pattern" | "codepattern" | "pattern" => Self::CodePattern,
            "decision" => Self::Decision,
            "constraint" => Self::Constraint,
            _ => Self::ProjectFact, // default
        }
    }

    fn topic_file_name(&self) -> &'static str {
        match self {
            Self::UserPreference => "user-preferences.md",
            Self::ProjectFact => "project-facts.md",
            Self::CodePattern => "code-patterns.md",
            Self::Decision => "decisions.md",
            Self::Constraint => "constraints.md",
        }
    }

    fn topic_title(&self) -> &'static str {
        match self {
            Self::UserPreference => "User Preferences",
            Self::ProjectFact => "Project Facts",
            Self::CodePattern => "Code Patterns",
            Self::Decision => "Decisions",
            Self::Constraint => "Constraints",
        }
    }

    fn topic_hook(&self) -> &'static str {
        match self {
            Self::UserPreference => "User workflow and communication preferences",
            Self::ProjectFact => "Stable facts about the project and stack",
            Self::CodePattern => "Reusable coding patterns and idioms",
            Self::Decision => "Decisions and rationale from past sessions",
            Self::Constraint => "Important constraints and non-negotiable limits",
        }
    }
}

/// A single fact extracted from the conversation.
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    /// The fact to remember, as a markdown bullet point or sentence.
    pub content: String,
    /// Semantic category for the fact.
    pub category: MemoryCategory,
    /// Model confidence, 0.0–1.0.
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Session state (tracks what has already been extracted)
// ---------------------------------------------------------------------------

/// Mutable per-session state for the memory extractor.
#[derive(Debug, Default)]
pub struct SessionMemoryState {
    /// UUID of the last message that was fully extracted.
    pub last_extracted_message_uuid: Option<String>,
    /// Token count at the time of the last extraction.
    pub tokens_at_last_extraction: u64,
    /// Whether session memory has been initialised for this session.
    pub initialized: bool,
}

impl SessionMemoryState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return `true` if the last extracted message UUID is still present in
    /// `messages`, which tells us new messages have been added since then.
    pub fn has_new_messages_since_last_extraction(&self, messages: &[Message]) -> bool {
        match &self.last_extracted_message_uuid {
            None => true, // Nothing extracted yet → treat all messages as new
            Some(uuid) => {
                messages
                    .iter()
                    .any(|m| m.uuid.as_deref() == Some(uuid.as_str()))
                    && messages.last().and_then(|m| m.uuid.as_deref()) != Some(uuid.as_str())
            }
        }
    }

    /// Advance the cursor to the last message in `messages`.
    pub fn advance_cursor(&mut self, messages: &[Message]) {
        self.last_extracted_message_uuid = messages.last().and_then(|m| m.uuid.clone());
    }
}

// ---------------------------------------------------------------------------
// SessionMemoryExtractor
// ---------------------------------------------------------------------------

/// Extracts and persists key memories from a conversation.
pub struct SessionMemoryExtractor {
    pub model: String,
    pub max_tokens: u32,
}

impl SessionMemoryExtractor {
    /// Create a new extractor using the given model.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: 4096,
        }
    }

    /// Return `true` if we have enough messages to warrant extraction.
    ///
    /// Mirrors `shouldExtractMemory` from TypeScript sessionMemory.ts:
    /// - At least `MIN_MESSAGES_TO_EXTRACT` messages total
    /// - The last assistant turn must not have pending tool calls (safe extraction point)
    pub fn should_extract(messages: &[Message]) -> bool {
        let model_visible = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::Assistant)
            .count();

        if model_visible < MIN_MESSAGES_TO_EXTRACT {
            return false;
        }

        // Don't extract mid-tool-call chain — wait for a clean end-turn
        let last_assistant = messages.iter().rev().find(|m| m.role == Role::Assistant);
        if let Some(last) = last_assistant {
            if last.has_tool_use() {
                return false; // still in a tool chain
            }
        }

        true
    }

    /// Count tool calls since the last extracted message UUID.
    fn count_tool_calls_since(messages: &[Message], since_uuid: Option<&str>) -> usize {
        let mut found_start = since_uuid.is_none();
        let mut count = 0usize;

        for msg in messages {
            if !found_start {
                if msg.uuid.as_deref() == since_uuid {
                    found_start = true;
                }
                continue;
            }
            if msg.role == Role::Assistant {
                count += msg.get_tool_use_blocks().len();
            }
        }
        count
    }

    /// Check whether extraction should run given the current session state.
    pub fn should_extract_with_state(messages: &[Message], state: &SessionMemoryState) -> bool {
        if !Self::should_extract(messages) {
            return false;
        }

        // Require minimum tool calls between updates (mirrors TS toolCallsBetweenUpdates)
        let tool_calls_since =
            Self::count_tool_calls_since(messages, state.last_extracted_message_uuid.as_deref());

        tool_calls_since >= MIN_TOOL_CALLS_BETWEEN_EXTRACTIONS
            || !state.has_new_messages_since_last_extraction(messages)
    }

    /// Extract key memories from a conversation.
    ///
    /// Calls the API with a structured extraction prompt and parses the
    /// response into `ExtractedMemory` entries.
    pub async fn extract(
        &self,
        messages: &[Message],
        working_dir: &Path,
        api_client: &mangocode_api::AnthropicClient,
    ) -> anyhow::Result<Vec<ExtractedMemory>> {
        let model_visible: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::Assistant)
            .collect();

        if model_visible.is_empty() {
            return Ok(vec![]);
        }

        // Build a compact transcript for the extraction prompt
        let mut transcript = String::new();
        for msg in &model_visible {
            let role_label = match msg.role {
                Role::User => "Human",
                Role::Assistant => "Assistant",
            };
            let text = msg.get_all_text();
            if !text.is_empty() {
                transcript.push_str(&format!("{}: {}\n\n", role_label, text));
            }
        }

        let working_dir_str = working_dir.display().to_string();
        let prompt = build_extraction_prompt(&transcript, &working_dir_str);

        let api_msgs = vec![ApiMessage {
            role: "user".to_string(),
            content: Value::String(prompt),
        }];

        let request = CreateMessageRequest::builder(&self.model, self.max_tokens)
            .messages(api_msgs)
            .system(SystemPrompt::Text(EXTRACTION_SYSTEM_PROMPT.to_string()))
            .build();

        let handler: Arc<dyn StreamHandler> = Arc::new(mangocode_api::streaming::NullStreamHandler);
        let mut rx = api_client
            .create_message_stream(request, handler)
            .await
            .map_err(|e| anyhow::anyhow!("API error: {}", e))?;

        let mut acc = StreamAccumulator::new();
        while let Some(evt) = rx.recv().await {
            acc.on_event(&evt);
            if matches!(evt, AnthropicStreamEvent::MessageStop) {
                break;
            }
        }

        let (response_msg, _usage, _stop) = acc.finish();
        let response_text = response_msg.get_all_text();

        if response_text.is_empty() {
            debug!("Session memory extraction produced empty response");
            return Ok(vec![]);
        }

        let memories = parse_extraction_response(&response_text);
        info!(count = memories.len(), "Session memory extraction complete");

        Ok(memories)
    }

    /// Persist extracted memories into the 3-layer memory hierarchy rooted at
    /// `memory_dir`:
    /// - layer 1 index: `MEMORY.md`
    /// - layer 2 topics: `*.md`
    /// - layer 3 transcripts: handled elsewhere (`conversations/*.jsonl`)
    pub async fn persist(memories: &[ExtractedMemory], memory_dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(memory_dir).await?;
        migrate_legacy_agents_if_needed(memory_dir).await?;

        if memories.is_empty() {
            return Ok(());
        }

        let mut grouped: BTreeMap<&'static str, (MemoryCategory, Vec<&ExtractedMemory>)> =
            BTreeMap::new();
        for memory in memories {
            let key = memory.category.topic_file_name();
            let entry = grouped
                .entry(key)
                .or_insert_with(|| (memory.category.clone(), Vec::new()));
            entry.1.push(memory);
        }

        let mut updated_topics: Vec<(MemoryCategory, String)> = Vec::new();
        let date_str = chrono::Local::now().format("%Y-%m-%d").to_string();
        for (topic_file, (category, mems)) in grouped {
            let path = memory_dir.join(topic_file);
            append_topic_memories(&path, &date_str, &mems).await?;
            updated_topics.push((category, topic_file.to_string()));
        }

        update_memory_index(memory_dir, &updated_topics).await?;
        info!(path = %memory_dir.display(), count = memories.len(), "Memories persisted");
        Ok(())
    }
}

pub async fn migrate_legacy_agents_if_needed(memory_dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(memory_dir).await?;

    let index_path = memory_dir.join("MEMORY.md");
    if path_exists(&index_path).await {
        return Ok(());
    }

    let Some(parent_dir) = memory_dir.parent() else {
        return Ok(());
    };
    let agents_path = parent_dir.join("AGENTS.md");
    if !path_exists(&agents_path).await {
        return Ok(());
    }

    let content = fs::read_to_string(&agents_path).await.unwrap_or_default();
    let legacy_memories = parse_legacy_agents_memories(&content);
    if !legacy_memories.is_empty() {
        let mut grouped: BTreeMap<&'static str, (MemoryCategory, Vec<&ExtractedMemory>)> =
            BTreeMap::new();
        for memory in &legacy_memories {
            let key = memory.category.topic_file_name();
            let entry = grouped
                .entry(key)
                .or_insert_with(|| (memory.category.clone(), Vec::new()));
            entry.1.push(memory);
        }

        let mut updated_topics: Vec<(MemoryCategory, String)> = Vec::new();
        let date_str = chrono::Local::now().format("%Y-%m-%d").to_string();
        for (topic_file, (category, mems)) in grouped {
            let path = memory_dir.join(topic_file);
            append_topic_memories(&path, &date_str, &mems).await?;
            updated_topics.push((category, topic_file.to_string()));
        }
        update_memory_index(memory_dir, &updated_topics).await?;
    } else {
        fs::write(&index_path, "").await?;
        if !content.trim().is_empty() {
            warn!(
                path = %agents_path.display(),
                "Legacy AGENTS.md format not recognized; leaving file in place"
            );
            return Ok(());
        }
    }

    let backup_path = parent_dir.join("AGENTS.md.bak");
    let _ = fs::rename(&agents_path, &backup_path).await;
    info!(
        path = %agents_path.display(),
        backup = %backup_path.display(),
        "Migrated legacy AGENTS.md to memory hierarchy"
    );

    Ok(())
}

async fn append_topic_memories(
    topic_path: &Path,
    date_str: &str,
    memories: &[&ExtractedMemory],
) -> anyhow::Result<()> {
    let mut existing = match fs::read_to_string(topic_path).await {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };

    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }

    existing.push_str(&format!("\n### {}\n", date_str));
    for memory in memories {
        existing.push_str(&format!(
            "- {} *(confidence: {:.0}%)*\n",
            memory.content,
            memory.confidence * 100.0
        ));
    }

    fs::write(topic_path, existing).await?;
    Ok(())
}

async fn update_memory_index(
    memory_dir: &Path,
    updated_topics: &[(MemoryCategory, String)],
) -> anyhow::Result<()> {
    let index_path = memory_dir.join("MEMORY.md");
    let existing = match fs::read_to_string(&index_path).await {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };

    let mut lines: Vec<String> = existing.lines().map(|l| l.to_string()).collect();
    for (category, topic_file) in updated_topics {
        let link = format!("memory/{}", topic_file);
        if !lines.iter().any(|line| line.contains(&format!("({})", link))) {
            lines.push(format!(
                "- [{}]({}) — {}",
                category.topic_title(),
                link,
                category.topic_hook()
            ));
        }
    }

    while lines.len() > 200 {
        if let Some(idx) = lines.iter().position(|line| {
            parse_index_topic_path(line)
                .map(|topic| memory_dir.join(topic).exists())
                .unwrap_or(false)
        }) {
            lines.remove(idx);
        } else {
            lines.remove(0);
        }
    }

    let mut out = lines.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    fs::write(index_path, out).await?;
    Ok(())
}

fn parse_index_topic_path(line: &str) -> Option<String> {
    let start = line.find("(memory/")?;
    let rest = &line[start + 8..];
    let end = rest.find(')')?;
    Some(rest[..end].to_string())
}

fn parse_legacy_agents_memories(content: &str) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if !line.starts_with("- **[") {
            continue;
        }

        let category = if line.contains("[user-preference]") {
            MemoryCategory::UserPreference
        } else if line.contains("[project-fact]") {
            MemoryCategory::ProjectFact
        } else if line.contains("[code-pattern]") {
            MemoryCategory::CodePattern
        } else if line.contains("[decision]") {
            MemoryCategory::Decision
        } else if line.contains("[constraint]") {
            MemoryCategory::Constraint
        } else {
            MemoryCategory::ProjectFact
        };

        let body = line
            .split("**")
            .nth(2)
            .map(str::trim)
            .unwrap_or("")
            .split("*(confidence")
            .next()
            .map(str::trim)
            .unwrap_or("")
            .to_string();

        if body.is_empty() {
            continue;
        }

        memories.push(ExtractedMemory {
            content: body,
            category,
            confidence: 0.7,
        });
    }
    memories
}

async fn path_exists(path: &Path) -> bool {
    fs::metadata(path).await.is_ok()
}

// ---------------------------------------------------------------------------
// Extraction prompt
// ---------------------------------------------------------------------------

const EXTRACTION_SYSTEM_PROMPT: &str = "You are a memory extraction assistant. Your job is to \
identify key facts, preferences, patterns, and decisions from a conversation that would be \
useful to remember for future interactions. Be precise, concise, and only extract genuinely \
useful information. Do not extract trivial or transient details.";

fn build_extraction_prompt(transcript: &str, working_dir: &str) -> String {
    format!(
        "Please analyze the following conversation transcript from a coding session in \
directory `{}` and extract key memories that would be useful to remember for future \
interactions.\n\
\n\
For each memory, output a line in this exact format:\n\
MEMORY: <category> | <confidence 0-10> | <concise fact>\n\
\n\
Where <category> is one of:\n\
- user_preference: how the user likes to work, communication style, tool preferences\n\
- project_fact: facts about the codebase, architecture, languages, frameworks\n\
- code_pattern: coding patterns, idioms, or styles used in this project\n\
- decision: key decisions made during the session\n\
- constraint: constraints, requirements, or limitations discovered\n\
\n\
Only output MEMORY: lines — no other text.  If there are no useful memories, output nothing.\n\
\n\
<conversation>\n\
{}\n\
</conversation>",
        working_dir, transcript
    )
}

/// Parse the model's response into `ExtractedMemory` entries.
/// Expected line format: `MEMORY: <category> | <confidence 0-10> | <fact>`
fn parse_extraction_response(response: &str) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();

    for line in response.lines() {
        let line = line.trim();
        if !line.starts_with("MEMORY:") {
            continue;
        }

        let rest = line["MEMORY:".len()..].trim();
        let parts: Vec<&str> = rest.splitn(3, '|').collect();
        if parts.len() < 3 {
            warn!("Skipping malformed memory line: {}", line);
            continue;
        }

        let category = MemoryCategory::from_str(parts[0].trim());
        let confidence_raw: f32 = parts[1].trim().parse().unwrap_or(5.0);
        // Normalize confidence from 0-10 scale to 0.0-1.0
        let confidence = (confidence_raw / 10.0).clamp(0.0, 1.0);
        let content = parts[2].trim().to_string();

        if content.is_empty() {
            continue;
        }

        memories.push(ExtractedMemory {
            content,
            category,
            confidence,
        });
    }

    memories
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::types::Message;

    fn make_user(text: &str) -> Message {
        Message::user(text)
    }

    fn make_assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    fn make_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    make_user(&format!("user message {}", i))
                } else {
                    make_assistant(&format!("assistant reply {}", i))
                }
            })
            .collect()
    }

    // ---- should_extract ------------------------------------------------

    #[test]
    fn test_should_not_extract_too_few_messages() {
        let msgs = make_messages(5);
        assert!(!SessionMemoryExtractor::should_extract(&msgs));
    }

    #[test]
    fn test_should_extract_enough_messages() {
        let msgs = make_messages(MIN_MESSAGES_TO_EXTRACT);
        // All messages are simple text (no tool use), so the last assistant
        // doesn't have pending tool calls — should be ok to extract.
        assert!(SessionMemoryExtractor::should_extract(&msgs));
    }

    #[test]
    fn test_should_extract_above_threshold() {
        let msgs = make_messages(MIN_MESSAGES_TO_EXTRACT + 1);
        assert!(SessionMemoryExtractor::should_extract(&msgs));
    }

    #[test]
    fn test_should_not_extract_mid_tool_chain() {
        use mangocode_core::types::ContentBlock;
        let mut msgs = make_messages(MIN_MESSAGES_TO_EXTRACT);
        // Replace the last assistant message with one that has a tool_use block
        let last = msgs.last_mut().unwrap();
        *last = Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        }]);
        // Last assistant has tool_use → extraction should be deferred
        assert!(!SessionMemoryExtractor::should_extract(&msgs));
    }

    // ---- parse_extraction_response -------------------------------------

    #[test]
    fn test_parse_empty_response() {
        let memories = parse_extraction_response("");
        assert!(memories.is_empty());
    }

    #[test]
    fn test_parse_single_memory() {
        let response = "MEMORY: project_fact | 8 | The project uses Rust 2021 edition";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].category, MemoryCategory::ProjectFact);
        assert!((memories[0].confidence - 0.8).abs() < 0.01);
        assert!(memories[0].content.contains("Rust 2021"));
    }

    #[test]
    fn test_parse_multiple_memories() {
        let response = "\
MEMORY: user_preference | 9 | User prefers verbose error messages\n\
MEMORY: decision | 7 | Decided to use tokio async runtime\n\
MEMORY: constraint | 6 | Must support Windows paths";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 3);
        assert_eq!(memories[0].category, MemoryCategory::UserPreference);
        assert_eq!(memories[1].category, MemoryCategory::Decision);
        assert_eq!(memories[2].category, MemoryCategory::Constraint);
    }

    #[test]
    fn test_parse_ignores_non_memory_lines() {
        let response = "Here are the memories:\n\
MEMORY: project_fact | 8 | Uses serde for JSON\n\
This is some extra text.\n\
MEMORY: code_pattern | 7 | Uses builder pattern";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_parse_malformed_line_skipped() {
        let response = "MEMORY: only_two_parts | no_confidence";
        let memories = parse_extraction_response(response);
        assert!(memories.is_empty());
    }

    #[test]
    fn test_parse_confidence_normalization() {
        let response = "MEMORY: decision | 10 | High confidence fact";
        let memories = parse_extraction_response(response);
        assert!((memories[0].confidence - 1.0).abs() < 0.01);
    }

    // ---- MemoryCategory parsing ----------------------------------------

    #[test]
    fn test_category_from_str_variants() {
        assert_eq!(
            MemoryCategory::from_str("user_preference"),
            MemoryCategory::UserPreference
        );
        assert_eq!(
            MemoryCategory::from_str("project_fact"),
            MemoryCategory::ProjectFact
        );
        assert_eq!(
            MemoryCategory::from_str("code_pattern"),
            MemoryCategory::CodePattern
        );
        assert_eq!(
            MemoryCategory::from_str("decision"),
            MemoryCategory::Decision
        );
        assert_eq!(
            MemoryCategory::from_str("constraint"),
            MemoryCategory::Constraint
        );
    }

    #[test]
    fn test_category_unknown_defaults_to_project_fact() {
        assert_eq!(
            MemoryCategory::from_str("totally_unknown"),
            MemoryCategory::ProjectFact
        );
    }

    // ---- persist (integration-ish with tempfile) -----------------------

    #[tokio::test]
    async fn test_persist_creates_topic_files_and_index() {
        let dir = tempfile::tempdir().unwrap();
        let memory_dir = dir.path().join(".mangocode").join("memory");

        let memories = vec![ExtractedMemory {
            content: "Uses async Rust".to_string(),
            category: MemoryCategory::ProjectFact,
            confidence: 0.9,
        }];

        SessionMemoryExtractor::persist(&memories, &memory_dir)
            .await
            .unwrap();

        let topic = fs::read_to_string(memory_dir.join("project-facts.md"))
            .await
            .unwrap();
        let index = fs::read_to_string(memory_dir.join("MEMORY.md")).await.unwrap();
        assert!(topic.contains("Uses async Rust"));
        assert!(index.contains("memory/project-facts.md"));
    }

    #[tokio::test]
    async fn test_persist_appends_to_existing_topic() {
        let dir = tempfile::tempdir().unwrap();
        let memory_dir = dir.path().join("memory");
        fs::create_dir_all(&memory_dir).await.unwrap();
        let topic_path = memory_dir.join("user-preferences.md");

        fs::write(&topic_path, "# Existing\n")
            .await
            .unwrap();

        let memories = vec![ExtractedMemory {
            content: "Prefers explicit error handling".to_string(),
            category: MemoryCategory::UserPreference,
            confidence: 0.8,
        }];

        SessionMemoryExtractor::persist(&memories, &memory_dir)
            .await
            .unwrap();

        let content = fs::read_to_string(&topic_path).await.unwrap();
        assert!(content.contains("# Existing"));
        assert!(content.contains("Prefers explicit error handling"));
    }

    #[tokio::test]
    async fn test_migration_from_legacy_agents() {
        let dir = tempfile::tempdir().unwrap();
        let dot_dir = dir.path().join(".mangocode");
        let memory_dir = dot_dir.join("memory");
        fs::create_dir_all(&dot_dir).await.unwrap();

        let legacy = "# Notes\n\n## Auto-extracted memories\n\n- **[project-fact]** Uses serde *(confidence: 80%)*\n";
        fs::write(dot_dir.join("AGENTS.md"), legacy).await.unwrap();

        migrate_legacy_agents_if_needed(&memory_dir).await.unwrap();

        let topic = fs::read_to_string(memory_dir.join("project-facts.md"))
            .await
            .unwrap();
        let index = fs::read_to_string(memory_dir.join("MEMORY.md")).await.unwrap();
        assert!(topic.contains("Uses serde"));
        assert!(index.contains("memory/project-facts.md"));
        assert!(dot_dir.join("AGENTS.md.bak").exists());
    }

    #[tokio::test]
    async fn test_migration_preserves_unrecognized_legacy_agents() {
        let dir = tempfile::tempdir().unwrap();
        let dot_dir = dir.path().join(".mangocode");
        let memory_dir = dot_dir.join("memory");
        fs::create_dir_all(&dot_dir).await.unwrap();

        // Valid legacy file may exist in custom formats; migration should not
        // move it away if we cannot parse any structured memory entries.
        fs::write(dot_dir.join("AGENTS.md"), "# Notes\n- plain bullet\n")
            .await
            .unwrap();

        migrate_legacy_agents_if_needed(&memory_dir).await.unwrap();

        assert!(dot_dir.join("AGENTS.md").exists());
        assert!(!dot_dir.join("AGENTS.md.bak").exists());
        assert!(memory_dir.join("MEMORY.md").exists());
    }

    #[tokio::test]
    async fn test_index_trim_to_200_lines() {
        let dir = tempfile::tempdir().unwrap();
        let memory_dir = dir.path().join("memory");
        fs::create_dir_all(&memory_dir).await.unwrap();

        let mut long_index = String::new();
        for i in 0..220 {
            let topic = format!("topic-{}.md", i);
            fs::write(memory_dir.join(&topic), "x").await.unwrap();
            long_index.push_str(&format!(
                "- [Topic {}](memory/{}) — hook {}\n",
                i, topic, i
            ));
        }
        fs::write(memory_dir.join("MEMORY.md"), long_index).await.unwrap();

        let memories = vec![ExtractedMemory {
            content: "constraint here".to_string(),
            category: MemoryCategory::Constraint,
            confidence: 0.8,
        }];

        SessionMemoryExtractor::persist(&memories, &memory_dir)
            .await
            .unwrap();

        let lines = fs::read_to_string(memory_dir.join("MEMORY.md"))
            .await
            .unwrap()
            .lines()
            .count();
        assert!(lines <= 200);
    }

    #[tokio::test]
    async fn test_persist_no_op_for_empty_memories() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("memory");

        SessionMemoryExtractor::persist(&[], &target).await.unwrap();

        // Directory may be created for migration checks, but no topic files exist
        assert!(!target.join("project-facts.md").exists());
    }

    // ---- SessionMemoryState --------------------------------------------

    #[test]
    fn test_state_has_new_messages_no_cursor() {
        let state = SessionMemoryState::new();
        let msgs = vec![make_user("Hello")];
        // No cursor → always has new messages
        assert!(state.has_new_messages_since_last_extraction(&msgs));
    }

    #[test]
    fn test_state_advance_cursor() {
        let mut state = SessionMemoryState::new();
        let mut msg = make_user("hello");
        msg.uuid = Some("uuid-1".to_string());
        let msgs = vec![msg];
        state.advance_cursor(&msgs);
        assert_eq!(state.last_extracted_message_uuid.as_deref(), Some("uuid-1"));
    }
}
