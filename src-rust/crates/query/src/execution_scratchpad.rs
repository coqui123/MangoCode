// execution_scratchpad.rs — Per-turn execution state tracking and injection.
//
// Implements the "execution scratchpad" layer described in the MangoCode hybrid
// architecture for long-horizon agentic sessions.  On every API call, a compact
// structured block is prepended to the dynamic system prompt so the model always
// has an explicit view of:
//
//   1. The current multi-step plan (what are we trying to accomplish overall)
//   2. The last tool result (one-line summary of the most recent tool output)
//   3. The declared next action (what the model said it would do next)
//
// This is provider-agnostic — it improves all models by giving deterministic
// context scaffolding and reducing the chance of goal-drift between turns.
//
// Enable/disable via feature flag FLAG_EXECUTION_SCRATCHPAD (default: on).
// The flag can be toggled at runtime via MANGOCODE_FLAG_EXECUTION_SCRATCHPAD=0.

use mangocode_core::types::{ContentBlock, Message, MessageContent, Role};

/// How many characters of a tool result to include in the one-line summary.
const RESULT_SUMMARY_CHARS: usize = 200;

/// Maximum length of the plan text stored in scratchpad state.
const MAX_PLAN_CHARS: usize = 500;

/// The scratchpad section header/footer used to delimit the injected block.
const SCRATCHPAD_HEADER: &str = "[SCRATCHPAD]";
const SCRATCHPAD_FOOTER: &str = "[/SCRATCHPAD]";

// ---------------------------------------------------------------------------
// ScratchpadState
// ---------------------------------------------------------------------------

/// Mutable per-session state for the execution scratchpad.
///
/// Updated after every turn based on the model's most recent output and
/// the tool results returned to it.
#[derive(Debug, Clone, Default)]
pub struct ScratchpadState {
    /// The current high-level plan for this session (extracted from model output).
    pub current_plan: Option<String>,

    /// One-line summary of the most recent tool call + result.
    pub last_tool_summary: Option<String>,

    /// The next declared action (extracted from end of last assistant turn).
    pub next_action: Option<String>,

    /// Turn counter at which the scratchpad was last updated.
    pub last_updated_turn: u32,
}

impl ScratchpadState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update state after a completed turn.
    ///
    /// `messages` is the full conversation history post-turn.
    /// `turn` is the current turn number.
    pub fn update_from_turn(&mut self, messages: &[Message], turn: u32) {
        self.last_updated_turn = turn;

        // Extract the last tool result summary.
        self.last_tool_summary = extract_last_tool_result_summary(messages);

        // Extract next-action hint from the last assistant text block.
        self.next_action = extract_next_action_hint(messages);
    }

    /// Update the current plan explicitly (called when model declares a plan).
    pub fn set_plan(&mut self, plan: impl Into<String>) {
        let p = plan.into();
        if p.len() > MAX_PLAN_CHARS {
            self.current_plan = Some(format!("{}…", &p[..MAX_PLAN_CHARS]));
        } else {
            self.current_plan = Some(p);
        }
    }

    /// Returns `true` if there is any scratchpad content worth injecting.
    pub fn has_content(&self) -> bool {
        self.current_plan.is_some()
            || self.last_tool_summary.is_some()
            || self.next_action.is_some()
    }

    /// Render the scratchpad as a text block suitable for prepending to the
    /// dynamic system prompt section or injecting as a synthetic user message.
    ///
    /// Returns `None` when the scratchpad is empty (first turn, no state yet).
    pub fn render(&self) -> Option<String> {
        if !self.has_content() {
            return None;
        }

        let mut lines = Vec::new();
        lines.push(SCRATCHPAD_HEADER.to_string());

        if let Some(ref plan) = self.current_plan {
            lines.push(format!("Plan: {}", plan));
        } else {
            lines.push("Plan: (not yet established)".to_string());
        }

        if let Some(ref tool) = self.last_tool_summary {
            lines.push(format!("Last tool: {}", tool));
        }

        if let Some(ref next) = self.next_action {
            lines.push(format!("Next step: {}", next));
        }

        lines.push(SCRATCHPAD_FOOTER.to_string());
        Some(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Extract a one-line summary from the most recent tool result in `messages`.
fn extract_last_tool_result_summary(messages: &[Message]) -> Option<String> {
    // Walk messages in reverse; find the last user message containing tool results.
    for msg in messages.iter().rev() {
        if msg.role != Role::User {
            continue;
        }
        let blocks = match &msg.content {
            MessageContent::Blocks(b) => b,
            _ => continue,
        };

        // Collect tool result text from this message.
        let mut result_texts: Vec<String> = Vec::new();
        for block in blocks {
            if let ContentBlock::ToolResult {
                tool_use_id, content, ..
            } = block
            {
                // Use a truncated tool_use_id as the label (first 8 chars) since
                // ToolResult does not carry a tool name — the name lives in the
                // corresponding ToolUse block. The id is sufficient for the scratchpad.
                let name_hint = if tool_use_id.len() > 8 {
                    &tool_use_id[..8]
                } else {
                    tool_use_id.as_str()
                };
                let name = name_hint;
                let text = match content {
                    mangocode_core::types::ToolResultContent::Text(t) => t.clone(),
                    mangocode_core::types::ToolResultContent::Blocks(inner) => inner
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                // One-line: strip newlines, truncate.
                let one_line = text.lines().next().unwrap_or("").trim().to_string();
                let summary = if one_line.len() > RESULT_SUMMARY_CHARS {
                    format!("{}…", &one_line[..RESULT_SUMMARY_CHARS])
                } else {
                    one_line
                };
                result_texts.push(format!("{}: {}", name, summary));
            }
        }

        if !result_texts.is_empty() {
            return Some(result_texts.join("; "));
        }
    }
    None
}

/// Extract a "next step" hint from the last assistant turn.
///
/// Looks for the last non-empty text sentence/line in the last assistant
/// message as a proxy for what the model declared it would do next.
fn extract_next_action_hint(messages: &[Message]) -> Option<String> {
    for msg in messages.iter().rev() {
        if msg.role != Role::Assistant {
            continue;
        }
        let text = match &msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        };

        // Find the last meaningful line (skip blank lines, tool XML, etc.)
        let last_line = text
            .lines()
            .rev()
            .find(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty()
                    && !trimmed.starts_with('<')
                    && !trimmed.starts_with("```")
                    && trimmed.len() > 10
            })
            .map(|l| l.trim().to_string());

        if let Some(line) = last_line {
            let hint = if line.len() > 150 {
                format!("{}…", &line[..150])
            } else {
                line
            };
            return Some(hint);
        }
        break; // only look at the most recent assistant turn
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::types::{Message, MessageContent, Role};

    fn text_msg(role: Role, text: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(text.to_string()),
            uuid: None,
            cost: None,
        }
    }

    #[test]
    fn render_returns_none_when_empty() {
        let state = ScratchpadState::new();
        assert!(state.render().is_none());
    }

    #[test]
    fn render_includes_all_fields() {
        let mut state = ScratchpadState::new();
        state.set_plan("Fix the failing tests in the auth module");
        state.last_tool_summary = Some("bash: exit code 0".to_string());
        state.next_action = Some("Run the test suite to verify the fix".to_string());

        let rendered = state.render().unwrap();
        assert!(rendered.contains(SCRATCHPAD_HEADER));
        assert!(rendered.contains("Plan:"));
        assert!(rendered.contains("Last tool:"));
        assert!(rendered.contains("Next step:"));
        assert!(rendered.contains(SCRATCHPAD_FOOTER));
    }

    #[test]
    fn extract_next_action_from_assistant_turn() {
        let messages = vec![
            text_msg(Role::User, "Fix the bug"),
            text_msg(
                Role::Assistant,
                "I'll start by reading the error logs.\nNext I will edit the source file.",
            ),
        ];
        let hint = extract_next_action_hint(&messages);
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("edit the source file"));
    }

    #[test]
    fn plan_is_truncated_at_max_chars() {
        let mut state = ScratchpadState::new();
        let long_plan = "x".repeat(MAX_PLAN_CHARS + 100);
        state.set_plan(&long_plan);
        let stored = state.current_plan.unwrap();
        // The ellipsis '\u{2026}' is 3 bytes in UTF-8, so stored.len() == MAX_PLAN_CHARS + 3.
        assert!(stored.len() <= MAX_PLAN_CHARS + 4);
        assert!(stored.ends_with('\u{2026}'));
    }
}
